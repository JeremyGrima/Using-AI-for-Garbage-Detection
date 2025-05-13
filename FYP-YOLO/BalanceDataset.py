import yaml
import shutil
from pathlib import Path
import random

# Path to your data.yaml:
yamlPath = Path("data", "data.yaml")  # Assuming it's in ./data/data.yaml

# Load data.yaml
with open(yamlPath, "r") as file:
    Dataset = yaml.safe_load(file)

# Extract class names and image paths from data.yaml
classNames = Dataset["names"]
trainingImages = Dataset["train"]
validationImages = Dataset["val"]
testingImages = Dataset["test"]

# Get the label directories by replacing "images" with "labels"
trainLabels = trainingImages.replace("images", "labels")
validationLabels = validationImages.replace("images", "labels")
testingLabels = testingImages.replace("images", "labels")

def balance_dataset(labelsDir, imagesDir):
    # Sampling more classes that are underrepresented, Physically duplicates the images and labels

    all_txt_files = list(Path(labelsDir).glob("*.txt"))
    # Debugging
    #print(f"\nDEBUG => Searching in {labelsDir}")
    #print(f"Found {len(all_txt_files)} .txt files:")

    # Count occurrences for each class
    class_counts = {name: 0 for name in classNames}

    for label_file in all_txt_files:
        with open(label_file, "r") as lf:
            for line in lf:
                class_id = int(line.split()[0])
                class_counts[classNames[class_id]] += 1

    print(f"Class distribution BEFORE balancing in {labelsDir}:")
    for cname, ccount in class_counts.items():
        print(f"  {cname}: {ccount}")

    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    # Threshold to decide if it should upsample
    THRESHOLD = 1.5
    if max_count > THRESHOLD * min_count:
        print("Imbalance detected. Balancing now.")

        for class_name, count in class_counts.items():
            if count < min_count * 1.2:  # 20% above the min
                print(f"Upsampling class: {class_name}")
                class_id = classNames.index(class_name)

                # Gather all .txt label files that reference this class_id
                label_files = []
                for f in Path(labelsDir).glob("*.txt"):
                    with open(f, "r") as lf:
                        lines = lf.read().split()
                        if str(class_id) in lines:
                            label_files.append(f)

                # The matching .jpg images
                image_files = [Path(imagesDir) / (f.stem + ".jpg") for f in label_files]

                if not image_files: #In case no images of a class appeared
                    print(f"No images found for class '{class_name}'. Skipping.")
                    continue

                while len(label_files) < min_count * 1.2:
                    src_img = random.choice(image_files)
                    src_label = Path(labelsDir) / (src_img.stem + ".txt")

                    rand_suffix = random.randint(1000,9999)
                    new_img = Path(imagesDir) / f"{src_img.stem}_copy{rand_suffix}.jpg"
                    new_label = Path(labelsDir) / f"{src_label.stem}_copy{rand_suffix}.txt"

                    shutil.copy(src_img, new_img)
                    shutil.copy(src_label, new_label)

                    label_files.append(new_label)
                    print(f"Duplicated {src_img.name} → {new_img.name}")

        print("Dataset balancing complete!")
    else:
        print("Dataset is already balanced. No action taken.")

# Run the balance function on each subset
balance_dataset(trainLabels, trainingImages)
balance_dataset(validationLabels, validationImages)
balance_dataset(testingLabels, testingImages)
