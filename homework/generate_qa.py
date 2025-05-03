import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    import math
    from pathlib import Path
    from PIL import Image
    import json

    # Load JSON info
    info_path = Path(info_path)
    with open(info_path) as f:
        info = json.load(f)

    # Construct and load image
    base_name = info_path.stem.replace("_info", "")
    image_file = info_path.parent / f"{base_name}_{view_index:02d}_im.jpg"
    img = Image.open(image_file)
    img_w, img_h = img.size

    # Compute scaling
    scale_x = img_w / ORIGINAL_WIDTH
    scale_y = img_h / ORIGINAL_HEIGHT

    karts = []
    detections = info.get("detections", [])
    frame_dets = detections[view_index] if view_index < len(detections) else []
    kart_names = info.get("kart_names", [])
    for det in frame_dets:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1:
            continue
        # Scale coords
        x1s, y1s = x1 * scale_x, y1 * scale_y
        x2s, y2s = x2 * scale_x, y2 * scale_y
        # Filter small/outside boxes
        if (x2s - x1s) < min_box_size or (y2s - y1s) < min_box_size:
            continue
        if x2s < 0 or x1s > img_w or y2s < 0 or y1s > img_h:
            continue
        cx, cy = (x1s + x2s) / 2, (y1s + y2s) / 2
        name = kart_names[track_id] if track_id < len(kart_names) else str(track_id)
        karts.append({"instance_id": track_id, "kart_name": name, "center": (cx, cy)})

    # Identify center kart by proximity to image center
    img_center = (img_w / 2, img_h / 2)
    ego_id, min_dist = None, float("inf")
    for obj in karts:
        dx, dy = obj["center"][0] - img_center[0], obj["center"][1] - img_center[1]
        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist, ego_id = dist, obj["instance_id"]
    for obj in karts:
        obj["is_center_kart"] = (obj["instance_id"] == ego_id)
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    import json
    from pathlib import Path

    info_path = Path(info_path)
    with open(info_path) as f:
        info = json.load(f)
    # Try common keys for track
    track = info.get("track_name") or info.get("track") or info.get("map_name") or info.get("map")
    if not track:
        raise KeyError("Track information not found in JSON")
    return str(track).lower().replace(" ", "_")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    from pathlib import Path

    # Generate QA pairs for one view
    karts = extract_kart_objects(info_path, view_index)
    track = extract_track_info(info_path)
    ego = next(obj for obj in karts if obj["is_center_kart"])
    ego_name = ego["kart_name"]
    ego_cx, ego_cy = ego["center"]

    qa_list = []
    # 1. Ego car identity
    qa_list.append({"question": "What kart is the ego car?", "answer": ego_name})
    # 2. Total karts
    qa_list.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})
    # 3. Track info
    qa_list.append({"question": "What track is this?", "answer": track})

    # 4. Relative and combined positions
    left_count = front_count = 0
    for obj in karts:
        if obj["is_center_kart"]:
            continue
        name = obj["kart_name"]
        cx, cy = obj["center"]
        lr = "left" if cx < ego_cx else "right"
        fb = "front" if cy < ego_cy else "behind"
        qa_list.append({"question": f"Is {name} to the left or right of the ego car?", "answer": lr})
        qa_list.append({"question": f"Is {name} in front of or behind the ego car?", "answer": fb})
        qa_list.append({"question": f"Where is {name} relative to the ego car?", "answer": f"{fb} and {lr}"})
        if lr == "left": left_count += 1
        if fb == "front": front_count += 1

    # 5. Aggregated counts
    qa_list.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count)})
    qa_list.append({"question": "How many karts are to the right of the ego car?", "answer": str(len(karts) - 1 - left_count)})
    qa_list.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count)})
    qa_list.append({"question": "How many karts are behind the ego car?", "answer": str(len(karts) - 1 - front_count)})

    # Attach image_file field
    split = Path(info_path).parent.name
    base = Path(info_path).stem.replace("_info", "")
    img_rel = f"{split}/{base}_{view_index:02d}_im.jpg"
    for e in qa_list:
        e["image_file"] = img_rel
    return qa_list


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(split: str = "train"):
    """Generate QA pairs for all info files in the specified split"""
    from pathlib import Path
    import json

    data_dir = Path(__file__).parent.parent / "data" / split
    all_qas = []
    for info_file in sorted(data_dir.glob("*_info.json")):
        with open(info_file) as f:
            info = json.load(f)
        num_views = len(info.get("detections", []))
        for vi in range(num_views):
            all_qas.extend(generate_qa_pairs(str(info_file), vi))
    output_file = data_dir / f"{split}_qa_pairs.json"
    with open(output_file, "w") as fw:
        json.dump(all_qas, fw, indent=2)
    print(f"Saved {len(all_qas)} QA pairs to {output_file}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate_all})


if __name__ == "__main__":
    main()
