#!/usr/bin/env python3
"""
Extract commenter types from regulations.gov comments for taxonomy building.

Usage:
    python extract_commenter_types.py <docket_id> [--sample-size 300]

Example:
    python extract_commenter_types.py EOIR-2020-0003 --sample-size 300
"""

import json
import time
import random
import os
import asyncio
from typing import List, Dict, Optional
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Import the main API client from reggie
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reggie.api import RegulationsAPIClient

# Load environment variables from .env file
load_dotenv()


async def sample_comments_from_docket(
    api_client: RegulationsAPIClient,
    docket_id: str,
    sample_size: int = 300
) -> List[Dict]:
    """Sample comments from a docket using the main API client.

    Args:
        api_client: RegulationsAPIClient instance
        docket_id: Docket ID to sample from
        sample_size: Number of comments to sample

    Returns:
        List of sampled comment dicts
    """
    print(f"Note: This is a simplified sampling implementation.")
    print(f"For full docket sampling, you would need to implement document listing.")
    print(f"Currently just returning empty list as placeholder.")
    # This would require implementing get_documents_for_docket in the main API client
    # For now, return empty list
    return []


class CommenterTypeExtractor:
    FEW_SHOT_EXAMPLES = [
        {
            "comment": "As a practicing neuropsychologist with 15 years of experience...",
            "output": "Neuropsychologist"
        },
        {
            "comment": "I am writing on behalf of the American Medical Association to express...",
            "output": "Professional Association (Medical)"
        },
        {
            "comment": "I'm an occupational therapist and I strongly oppose this regulation...",
            "output": "Occupational Therapist"
        }
    ]
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = OpenAI(api_key=openai_api_key)
    
    def extract_commenter_type(self, comment_text: str) -> str:
        """Extract commenter type from a comment using GPT-5-nano with few-shot prompting."""
        
        # Build few-shot prompt
        examples_text = "\n\n".join([
            f"Comment: \"{ex['comment']}\"\nCommenter Type: {ex['output']}"
            for ex in self.FEW_SHOT_EXAMPLES
        ])
        
        prompt = f"""You are analyzing public comments on healthcare regulations. Extract only a description of the commenter.

Examples:
{examples_text}

Now extract the commenter description from this comment:
Comment: "{comment_text}"

Commenter Type:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You extract commenter types from regulatory comments. Respond with ONLY the commenter type, nothing else."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            commenter_type = response.choices[0].message.content.strip()
            
            return commenter_type
            
        except Exception as e:
            print(f"Error extracting commenter type: {e}")
            return "Error - Unable to Extract"
    
    async def extract_batch(self, comments: List[Dict], api_client: RegulationsAPIClient) -> List[str]:
        """Extract commenter types from a batch of comments.

        Args:
            comments: List of comment dicts
            api_client: RegulationsAPIClient instance

        Returns:
            List of extracted commenter types
        """
        results = []

        for i, comment in enumerate(comments):
            comment_id = comment.get("id", "unknown")

            print(f"Processing comment {i+1}/{len(comments)} ({comment_id})...")
            print(f"  Fetching full comment details...")

            # Fetch the full comment details to get complete information
            try:
                comment_details = await api_client.get_comment_details(comment_id)
                attributes = comment_details.get("attributes", {})
            except Exception as e:
                print(f"  Error fetching comment details: {e}")
                attributes = comment.get("attributes", {})

            # Get comment text
            comment_text = attributes.get("comment", "")

            # Try to get organization/name from metadata
            first_name = attributes.get("firstName", "")
            last_name = attributes.get("lastName", "")
            organization = attributes.get("organization", "")

            # Build context
            full_text = f"{organization}. {first_name} {last_name}. {comment_text}"

            print(f"  Comment preview: {full_text[:150]}...")

            commenter_type = self.extract_commenter_type(full_text)

            results.append(commenter_type)

            await asyncio.sleep(0.2)  # Rate limiting for OpenAI

        return results


async def async_main():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Extract commenter types from regulations.gov for taxonomy building"
    )
    parser.add_argument("docket_id", help="Docket ID (e.g., EOIR-2020-0003)")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of comments to sample")
    parser.add_argument("--output", default="commenter_types.json", help="Output JSON file")

    args = parser.parse_args()

    # Get OpenAI API key from environment variable
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Get Regulations.gov API key from environment variable, fallback to DEMO_KEY
    regs_api_key = os.getenv("REG_API_KEY", "DEMO_KEY")

    # Initialize clients
    api_client = RegulationsAPIClient(api_key=regs_api_key)
    extractor = CommenterTypeExtractor(openai_api_key=openai_key)

    # Sample comments
    print(f"\n{'='*60}")
    print(f"Sampling {args.sample_size} comments from docket {args.docket_id}")
    print(f"{'='*60}\n")

    comments = await sample_comments_from_docket(api_client, args.docket_id, args.sample_size)

    if not comments:
        print("No comments sampled. Exiting.")
        await api_client.close()
        return

    # Extract commenter types
    print(f"\n{'='*60}")
    print(f"Extracting commenter types using GPT-5-nano")
    print(f"{'='*60}\n")

    results = await extractor.extract_batch(comments, api_client)

    await api_client.close()

    # Delete old output file if it exists
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"\nDeleted old output file: {args.output}")

    # Save results
    output_data = {
        "docket_id": args.docket_id,
        "sample_size": len(results),
        "commenter_types": results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}\n")

    from collections import Counter
    type_counts = Counter(results)

    print(f"Total comments analyzed: {len(results)}")
    print(f"\nTop 20 commenter types:")
    for commenter_type, count in type_counts.most_common(20):
        print(f"  {count:4d}  {commenter_type}")

    print(f"\nFull results saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the commenter types")
    print(f"  2. Group similar types into categories")
    print(f"  3. Define your final taxonomy")


def main():
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()