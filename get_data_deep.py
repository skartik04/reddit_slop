import requests
import json
import time
import datetime
import random
import signal
import sys

# --- CONFIGURATION ---
SUBREDDIT = "benignexistence"
TARGET_COUNT = 600  # We aim for this many total examples
OUTPUT_FILE = "benign_existence_deep_data.json"
SAVE_EVERY = 10  # Auto-save every N items collected
# ---------------------

# Global dataset for signal handler access
dataset = []
last_saved_count = 0  # Track when we last saved

def save_progress(reason="manual"):
    """Save current progress to file."""
    if dataset:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"\n💾 Saved {len(dataset)} examples ({reason})")
    else:
        print(f"\n⚠️ No data to save yet ({reason})")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by saving progress."""
    print("\n\n🛑 Interrupted! Saving progress...")
    save_progress("interrupted")
    sys.exit(0)

# Register signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Pool of realistic User-Agents to rotate through
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Mobile - iPhone
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    # Mobile - Android
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
]

def get_random_headers():
    """Returns headers with a random User-Agent."""
    return {"User-Agent": random.choice(USER_AGENTS)}

def get_comments_for_post(post_id, max_retries=5):
    """Fetches the top comment (solution) for a post. Retries on rate limits, then skips."""
    url = f"https://www.reddit.com/comments/{post_id}.json"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_random_headers())
            
            if response.status_code == 429:
                save_progress("rate limited - comment fetch")
                wait_time = random.uniform(60, 120)  # Wait longer overnight
                print(f"   Rate limited. Waiting {wait_time:.0f}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200: 
                return None, 0
                
            data = response.json()
            comments = data[1]['data']['children']
            
            # Look for the best answer
            for comment in comments:
                c_data = comment['data']
                # Filter out deleted/removed/bot comments
                if c_data.get('stickied', False) or c_data.get('body') in ['[deleted]', '[removed]']:
                    continue
                return c_data['body'], c_data['score']
            return None, 0  # No valid comments found
            
        except Exception as e:
            print(f"   Comment fetch failed: {e}")
            time.sleep(10)
            continue
    
    print(f"   Skipping post {post_id} after {max_retries} retries")
    return None, 0

def main():
    global dataset, last_saved_count
    dataset = []
    seen_ids = set() 
    
    # We cycle through different "Views" to find hidden posts
    # "top" with different time windows is the best way to get volume
    strategies = [
        ("top", "all"),   # The absolute best
        ("top", "year"),  # Best of last year
        ("top", "month"), # Best of last month
        ("hot", None),    # Currently trending
        ("rising", None), # Up and coming
        ("new", None)     # Brand new (mixed quality, requires filtering)
    ]

    print(f"🚀 Deep Scraping r/{SUBREDDIT} to reach {TARGET_COUNT} examples...")
    print(f"   Auto-saves every {SAVE_EVERY} items. Press Ctrl+C to save and exit.")

    for sort_type, time_filter in strategies:
        if len(dataset) >= TARGET_COUNT:
            break
            
        print(f"\n--- Switching strategy: {sort_type} " + (f"(t={time_filter})" if time_filter else "") + " ---")
        
        after_token = None
        
        # Scrape up to 10 pages per strategy
        page = 0
        retries = 0
        while page < 10:
            if len(dataset) >= TARGET_COUNT: break
            
            # Construct URL
            url = f"https://www.reddit.com/r/{SUBREDDIT}/{sort_type}.json?limit=100"
            if time_filter: url += f"&t={time_filter}"
            if after_token: url += f"&after={after_token}"
            
            try:
                response = requests.get(url, headers=get_random_headers())
                
                if response.status_code == 429:
                    save_progress("rate limited - listing")
                    retries += 1
                    if retries > 5:
                        print("Too many rate limits for this strategy. Moving to next...")
                        break
                    wait_time = random.uniform(60, 120)
                    print(f"Rate limited. Waiting {wait_time:.0f}s... (retry {retries}/5)")
                    time.sleep(wait_time)
                    continue  # Retry same page
                
                if response.status_code != 200: 
                    print(f"   Error {response.status_code}. Breaking.")
                    break
                
                retries = 0  # Reset retries on success
                data = response.json()
                posts = data['data']['children']
            except Exception as e:
                print(f"   Request failed: {e}")
                save_progress("request failed")
                time.sleep(30)
                continue  # Retry

            if not posts: break
            
            # Process Batch
            new_in_batch = 0
            for post in posts:
                p = post['data']
                
                # 1. Deduplicate
                if p['id'] in seen_ids: continue
                seen_ids.add(p['id'])
                
                # 2. Quality Control (Skip posts with no comments)
                if p['num_comments'] < 1: continue
                
                # 3. Filter for "Reasoning" quality BEFORE fetching comments
                # If scraping 'new', ensure it has at least 3 upvotes so it's not spam
                if sort_type == "new" and p['score'] < 3:
                    continue
                
                # 4. Fetch Solution (The Slow Part - only for qualified posts)
                time.sleep(random.uniform(0.8, 2.0))  # Randomized delay to look human
                solution, score = get_comments_for_post(p['id'])

                if solution:
                    dataset.append({
                        "id": p['id'],
                        "date": datetime.datetime.fromtimestamp(p['created_utc']).strftime('%Y-%m-%d'),
                        "instruction": "Diagnose the following dimensional anomaly.",
                        "input": f"{p['title']}\n{p['selftext']}".strip(),
                        "output": solution,
                        "upvotes": p['score']
                    })
                    new_in_batch += 1
                    print(f"   [{len(dataset)}/{TARGET_COUNT}] Found: {p['title'][:40]}...")
                    
                    # Auto-save every SAVE_EVERY items
                    if len(dataset) - last_saved_count >= SAVE_EVERY:
                        save_progress("periodic")
                        last_saved_count = len(dataset)
                    
                    # Stop immediately when we hit the target
                    if len(dataset) >= TARGET_COUNT:
                        break
            
            # Exit page loop if target reached
            if len(dataset) >= TARGET_COUNT:
                break

            # Advance Page
            after_token = data['data']['after']
            if not after_token: break
            
            print(f"   Page {page+1} done. Found {new_in_batch} new posts.")
            page += 1
            time.sleep(random.uniform(1.5, 3.5))  # Randomized pause between pages

    # Save Final
    save_progress("completed")
    print(f"\n✅ DONE! Total unique examples: {len(dataset)}")

if __name__ == "__main__":
    main()