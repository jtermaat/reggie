#!/usr/bin/env python3
"""
Extract topic/issue tags from regulations.gov comments for taxonomy building.

Usage:
    python comment_sample_analysis.py <docket_id> [--sample-size 300]

Example:
    python comment_sample_analysis.py EOIR-2020-0003 --sample-size 300
"""

import json
import random
import os
import asyncio
from typing import List, Dict, Optional
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Import database components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg
from reggie.db.connection import get_connection_string
from reggie.db.repository import CommentRepository

# Load environment variables from .env file
load_dotenv()


async def sample_comments_from_database(
    sample_size: int = 300,
    document_id: Optional[str] = None
) -> List[Dict]:
    """Sample comments from database.

    Args:
        sample_size: Number of comments to sample
        document_id: Optional document ID to filter by. If None, samples from all comments.

    Returns:
        List of sampled comment dicts with keys: id, comment_text, first_name, last_name, organization
    """
    connection_string = get_connection_string()

    if document_id:
        print(f"Fetching comments from database for document {document_id}...")
        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            comment_rows = await CommentRepository.get_comments_for_document(document_id, conn)
    else:
        print(f"Fetching comments from database (all documents)...")
        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, comment_text, first_name, last_name, organization
                    FROM comments
                    ORDER BY created_at
                    """
                )
                comment_rows = await cur.fetchall()

    print(f"Found {len(comment_rows)} total comments in database")

    if not comment_rows:
        print("No comments found.")
        return []

    # Convert CommentData objects to dict format
    all_comments = []
    for comment_data in comment_rows:
        comment_dict = {
            "id": comment_data.id,
            "comment_text": comment_data.comment_text,
            "first_name": comment_data.first_name,
            "last_name": comment_data.last_name,
            "organization": comment_data.organization
        }
        all_comments.append(comment_dict)

    # Sample randomly
    if len(all_comments) <= sample_size:
        sampled_comments = all_comments
        print(f"Using all {len(sampled_comments)} comments (less than sample size)")
    else:
        sampled_comments = random.sample(all_comments, sample_size)
        print(f"Randomly sampled {len(sampled_comments)} comments")

    return sampled_comments


class TopicIssueExtractor:
    FEW_SHOT_EXAMPLES = [
        {
            "comment": "As a practicing neuropsychologist, I'm concerned about the impact of these changes on patient privacy and the burden of increased documentation requirements. The proposed timelines are unrealistic.",
            "output": ["Patient Privacy", "Administrative Burden", "Implementation Timeline"]
        },
        {
            "comment": "I'm an occupational therapist and I strongly oppose this regulation because it will reduce access to care in rural areas and create unfair licensing requirements across state lines.",
            "output": ["Access to Care", "Rural Healthcare", "Licensing Requirements"]
        }
    ]

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = OpenAI(api_key=openai_api_key)

    def extract_topic_tags(self, comment_text: str) -> List[str]:
        """Extract topic/issue tags from a comment using GPT-5-nano with few-shot prompting."""

        # Build few-shot prompt
        examples_text = "\n\n".join([
            f"Comment: \"{ex['comment']}\"\nTopics/Issues: {json.dumps(ex['output'])}"
            for ex in self.FEW_SHOT_EXAMPLES
        ])

        prompt = f"""You are analyzing public comments on regulations. Extract a list of specific topics or issues discussed in the comment.

Each topic should be a concise phrase (2-4 words). Return 1-5 topics per comment, focusing on the main issues raised.

Examples:
{examples_text}

Now extract the topics/issues from this comment:
Comment: "{comment_text}"

Topics/Issues (respond with ONLY a JSON array of strings):"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You extract topic/issue tags from regulatory comments. Respond with ONLY a JSON array of topic strings, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()

            # Parse the JSON response
            try:
                # Try to parse as JSON object first (in case it's wrapped)
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    # If it's a dict, look for common keys
                    topics = parsed.get("topics") or parsed.get("issues") or parsed.get("tags") or []
                elif isinstance(parsed, list):
                    topics = parsed
                else:
                    topics = []

                # Ensure all items are strings
                topics = [str(t) for t in topics if t]

                return topics if topics else ["Unable to Extract"]

            except json.JSONDecodeError:
                # Fallback: try to parse as plain list
                return ["Unable to Extract"]

        except Exception as e:
            print(f"Error extracting topics: {e}")
            return ["Error - Unable to Extract"]
    
    async def extract_batch(self, comments: List[Dict]) -> List[List[str]]:
        """Extract topic/issue tags from a batch of comments.

        Args:
            comments: List of comment dicts with keys: id, comment_text, first_name, last_name, organization

        Returns:
            List of lists of extracted topic tags (one list per comment)
        """
        results = []

        for i, comment in enumerate(comments):
            comment_id = comment.get("id", "unknown")

            print(f"Processing comment {i+1}/{len(comments)} ({comment_id})...")

            # Get comment text from database format
            comment_text = comment.get("comment_text", "")

            print(f"  Comment preview: {comment_text[:150]}...")

            topic_tags = self.extract_topic_tags(comment_text)

            print(f"  Extracted topics: {topic_tags}")

            results.append(topic_tags)

            await asyncio.sleep(0.2)  # Rate limiting for OpenAI

        return results


async def async_main():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Extract topic/issue tags from regulations.gov for taxonomy building"
    )
    parser.add_argument("--document-id", help="Optional document ID to filter by (e.g., CMS-2025-0304-0009)")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of comments to sample")
    parser.add_argument("--output", default="topic_tags.json", help="Output JSON file")

    args = parser.parse_args()

    # Get OpenAI API key from environment variable
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize extractor
    extractor = TopicIssueExtractor(openai_api_key=openai_key)

    # Sample comments from database
    print(f"\n{'='*60}")
    if args.document_id:
        print(f"Sampling {args.sample_size} comments from document {args.document_id}")
    else:
        print(f"Sampling {args.sample_size} comments from all documents")
    print(f"{'='*60}\n")

    comments = await sample_comments_from_database(args.sample_size, args.document_id)

    if not comments:
        print("No comments sampled. Exiting.")
        return

    # Extract topic tags
    print(f"\n{'='*60}")
    print(f"Extracting topic/issue tags using GPT-5-nano")
    print(f"{'='*60}\n")

    results = await extractor.extract_batch(comments)

    # Delete old output file if it exists
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"\nDeleted old output file: {args.output}")

    # Save results
    output_data = {
        "document_id": args.document_id,
        "sample_size": len(results),
        "topic_tags": results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}\n")

    from collections import Counter

    # Flatten all tags to count frequencies
    all_tags = [tag for tag_list in results for tag in tag_list]
    tag_counts = Counter(all_tags)

    print(f"Total comments analyzed: {len(results)}")
    print(f"Total unique tags: {len(tag_counts)}")
    print(f"\nTop 30 most common topic/issue tags:")
    for tag, count in tag_counts.most_common(30):
        print(f"  {count:4d}  {tag}")

    print(f"\nFull results saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the topic/issue tags")
    print(f"  2. Group similar tags into categories")
    print(f"  3. Define your final topic taxonomy")


def main():
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()