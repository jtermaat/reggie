#!/usr/bin/env python3
"""
Extract commenter types from regulations.gov comments for taxonomy building.

Usage:
    python extract_commenter_types.py <docket_id> [--sample-size 300]

Example:
    python extract_commenter_types.py EOIR-2020-0003 --sample-size 300
"""

import requests
import json
import time
import random
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import argparse
from openai import OpenAI


@dataclass
class CommenterInfo:
    comment_id: str
    commenter_type: str
    confidence: str
    raw_comment_snippet: str


class RegulationsAPIClient:
    BASE_URL = "https://api.regulations.gov/v4"
    
    def __init__(self, api_key: str = "DEMO_KEY"):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key})
    
    def get_documents_for_docket(self, docket_id: str) -> List[Dict]:
        """Get all documents in a docket."""
        url = f"{self.BASE_URL}/documents"
        params = {"filter[docketId]": docket_id, "page[size]": 250}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return data.get("data", [])
    
    def get_comments_for_document(
        self, 
        object_id: str, 
        page_number: int = 1, 
        page_size: int = 250
    ) -> Dict:
        """Get comments for a specific document."""
        url = f"{self.BASE_URL}/comments"
        params = {
            "filter[commentOnId]": object_id,
            "page[size]": page_size,
            "page[number]": page_number,
            "sort": "lastModifiedDate,documentId"
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def sample_comments_from_docket(
        self, 
        docket_id: str, 
        sample_size: int = 300
    ) -> List[Dict]:
        """Sample comments from a docket."""
        print(f"Fetching documents for docket {docket_id}...")
        documents = self.get_documents_for_docket(docket_id)
        
        if not documents:
            raise ValueError(f"No documents found for docket {docket_id}")
        
        print(f"Found {len(documents)} documents in docket")
        
        all_comment_ids = []
        
        # Get comment counts for each document
        for doc in documents:
            object_id = doc.get("attributes", {}).get("objectId")
            if not object_id:
                continue
            
            print(f"Checking comments for document {doc['id']}...")
            response = self.get_comments_for_document(object_id, page_number=1, page_size=1)
            
            total_comments = response.get("meta", {}).get("totalElements", 0)
            if total_comments > 0:
                print(f"  Found {total_comments} comments")
                all_comment_ids.append({
                    "object_id": object_id,
                    "document_id": doc["id"],
                    "total_comments": total_comments
                })
            
            time.sleep(0.5)  # Rate limiting
        
        if not all_comment_ids:
            raise ValueError("No comments found in any documents")
        
        # Calculate sampling strategy
        total_comments = sum(d["total_comments"] for d in all_comment_ids)
        print(f"\nTotal comments in docket: {total_comments}")
        
        # Sample proportionally from each document
        sampled_comments = []
        for doc_info in all_comment_ids:
            proportion = doc_info["total_comments"] / total_comments
            doc_sample_size = max(1, int(sample_size * proportion))
            
            # Fetch random pages to get diverse samples
            max_pages = min(20, (doc_info["total_comments"] + 249) // 250)
            pages_to_fetch = min(3, max_pages)
            
            for _ in range(pages_to_fetch):
                page = random.randint(1, max_pages)
                response = self.get_comments_for_document(
                    doc_info["object_id"], 
                    page_number=page,
                    page_size=250
                )
                
                comments = response.get("data", [])
                sampled_comments.extend(comments)
                
                time.sleep(0.5)
                
                if len(sampled_comments) >= sample_size:
                    break
            
            if len(sampled_comments) >= sample_size:
                break
        
        # Randomly sample down to exact size
        if len(sampled_comments) > sample_size:
            sampled_comments = random.sample(sampled_comments, sample_size)
        
        print(f"\nSampled {len(sampled_comments)} comments")
        return sampled_comments


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
    
    def extract_commenter_type(self, comment_text: str) -> CommenterInfo:
        """Extract commenter type from a comment using GPT-4o-mini with few-shot prompting."""
        
        # Build few-shot prompt
        examples_text = "\n\n".join([
            f"Comment: \"{ex['comment']}\"\nCommenter Type: {ex['output']}"
            for ex in self.FEW_SHOT_EXAMPLES
        ])
        
        prompt = f"""You are analyzing public comments on healthcare regulations. Extract ONLY the professional role, credentials, or organizational affiliation of the commenter. Focus on what type of stakeholder they are.

Examples:
{examples_text}

Now extract the commenter type from this comment:
Comment: "{comment_text[:500]}"

Instructions:
- Extract ONLY who/what the commenter is (their role, profession, or organization type)
- Be specific about medical specialties when mentioned
- Distinguish between individuals, organizations, and advocacy groups
- If multiple roles mentioned, pick the most relevant to healthcare
- If no clear role is stated, write "Individual (Role Unspecified)"
- Keep your response under 10 words
- Do not include any other text, just the commenter type

Commenter Type:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract commenter types from regulatory comments. Respond with ONLY the commenter type, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=30
            )
            
            commenter_type = response.choices[0].message.content.strip()
            
            return commenter_type
            
        except Exception as e:
            print(f"Error extracting commenter type: {e}")
            return "Error - Unable to Extract"
    
    def extract_batch(self, comments: List[Dict]) -> List[CommenterInfo]:
        """Extract commenter types from a batch of comments."""
        results = []
        
        for i, comment in enumerate(comments):
            comment_id = comment.get("id", "unknown")
            attributes = comment.get("attributes", {})
            
            # Get comment text
            comment_text = attributes.get("comment", "")
            
            # Try to get organization/name from metadata
            first_name = attributes.get("firstName", "")
            last_name = attributes.get("lastName", "")
            organization = attributes.get("organization", "")
            
            # Build context
            full_text = f"{organization}. {first_name} {last_name}. {comment_text}"
            
            print(f"Processing comment {i+1}/{len(comments)} ({comment_id})...")
            
            commenter_type = self.extract_commenter_type(full_text)
            
            info = CommenterInfo(
                comment_id=comment_id,
                commenter_type=commenter_type,
                confidence="high" if any(keyword in full_text.lower()[:200] for keyword in ["i am a", "as a", "our organization"]) else "medium",
                raw_comment_snippet=full_text[:200]
            )
            
            results.append(info)
            
            time.sleep(0.2)  # Rate limiting for OpenAI
        
        return results


def main():
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
    regs_client = RegulationsAPIClient(api_key=regs_api_key)
    extractor = CommenterTypeExtractor(openai_api_key=openai_key)
    
    # Sample comments
    print(f"\n{'='*60}")
    print(f"Sampling {args.sample_size} comments from docket {args.docket_id}")
    print(f"{'='*60}\n")
    
    comments = regs_client.sample_comments_from_docket(args.docket_id, args.sample_size)
    
    # Extract commenter types
    print(f"\n{'='*60}")
    print(f"Extracting commenter types using GPT-4o-mini")
    print(f"{'='*60}\n")
    
    results = extractor.extract_batch(comments)
    
    # Save results
    output_data = {
        "docket_id": args.docket_id,
        "sample_size": len(results),
        "commenter_types": [
            {
                "comment_id": r.comment_id,
                "commenter_type": r.commenter_type,
                "confidence": r.confidence,
                "snippet": r.raw_comment_snippet
            }
            for r in results
        ]
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}\n")
    
    from collections import Counter
    type_counts = Counter(r.commenter_type for r in results)
    
    print(f"Total comments analyzed: {len(results)}")
    print(f"\nTop 20 commenter types:")
    for commenter_type, count in type_counts.most_common(20):
        print(f"  {count:4d}  {commenter_type}")
    
    print(f"\nFull results saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the commenter types")
    print(f"  2. Group similar types into categories")
    print(f"  3. Define your final taxonomy")


if __name__ == "__main__":
    main()