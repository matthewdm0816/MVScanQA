import os
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessExecutor
from openai import OpenAI
from PIL import Image
import io
import json
from tqdm.auto import tqdm
from glob import glob
import multiprocessing


def load_image(image_path):
    """Load an image and convert it to base64 encoding."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        # resize to max 512 height
        # aspect_ratio = img.width / img.height
        # new_height = min(512, img.height)
        # new_width = int(new_height * aspect_ratio)
        # img = img.resize((new_width, new_height), Image.LANCZOS)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
def get_caption(client, image_base64):
    """Get a caption for an image using GPT-4 Vision."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 60 words."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=512,
    )
    # Get the generated content
    content = response.choices[0].message.content.strip()

    # Get the number of tokens used
    tokens_used = response.usage.total_tokens

    return content, tokens_used

def process_image(client, image_path):
    """Process a single image: load it, get its caption, and return the result."""
    try:
        image_base64 = load_image(image_path)
        caption, tokens_used = get_caption(client, image_base64)
        return image_path, caption, tokens_used
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return image_path, f"Error: {str(e)}", 0
    
def format_large_integer(number):
    """Format a large integer with commas for better readability."""
    return f"{number / 1000:.1f}K" if number < 1000_000 else f"{number / 1000_000:.1f}M"

def caption_images(directory, api_key, num_workers=4, scene_range=(0, -1)):
    """Caption all images in the given directory using multiple worker threads."""
    client = OpenAI(
        api_key=api_key, # 
        base_url="https://api.openai.com/v1",
    )
    image_paths = []
    
    # Collect all image file paths
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_paths.append(os.path.join(root, file))
                # image_paths.append(file)

    scene_list = sorted(glob(f"{directory}/scene*"))
    print(f"Total scenes: {len(scene_list)}")
    scene_list = scene_list[scene_range[0]:scene_range[1]]
    print(f"Selected scenes: {scene_range[0]}-{scene_range[1]}")

    image_paths = []
    for scene in scene_list:
        scene_image_paths = sorted(glob(os.path.join(scene, 'color', "*.jpg")))
        image_paths.extend(scene_image_paths)

    print(f"Found {len(image_paths)} images")
    
    results = {}
    all_tokens_used = 0
    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the executor
        future_to_path = {executor.submit(process_image, client, path): path for path in image_paths}
        
        # Process completed futures as they finish
        # for future in as_completed(future_to_path):
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
            path = future_to_path[future]
            try:
                path, caption, tokens_used = future.result()
                path = os.path.relpath(path, directory)
                all_tokens_used += tokens_used
                formatted_tokens = f"{format_large_integer(all_tokens_used)} tokens"
                print(f"Tokens used: {formatted_tokens}")
                results[path] = caption
                print(f"Processed: {path}")
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Cancelling pending tasks...")
                for future in future_to_path:
                    future.cancel()
                executor.shutdown(wait=False)
                print("Exiting...")
                return results
            
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

    
    return results

def save_captions(results, output_file):
    """Save the image captions to a file."""

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """Main function to parse arguments and run the captioning process."""
    parser = argparse.ArgumentParser(description="Caption images using GPT-4 Vision")
    parser.add_argument("--directory", help="Directory containing images to caption")
    parser.add_argument("--api_key", help="OpenAI API key", required=True)
    parser.add_argument("--output", default="api-captions/scannet-captions-gpt4o.json", help="Output file for captions")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    parser.add_argument("--scene_range", type=int, nargs=2, default=[0, -1], help="Range of scenes to process")
    args = parser.parse_args()

    results = caption_images(args.directory, args.api_key, args.workers, args.scene_range)
    save_captions(results, args.output)
    print(f"Captions saved to {args.output}")

if __name__ == "__main__":
    main()
    