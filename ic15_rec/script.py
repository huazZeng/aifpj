import random

def split_txt_file(input_file, train_file, valid_file, valid_ratio=0.1, seed=42):
    """
    Splits a text file into training and validation files.
    
    Args:
        input_file (str): Path to the input text file.
        train_file (str): Path to save the training file.
        valid_file (str): Path to save the validation file.
        valid_ratio (float): Ratio of lines to use as the validation set.
        seed (int): Random seed for reproducibility.
    """
    # Load the input file
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Shuffle the lines
    random.seed(seed)
    random.shuffle(lines)
    
    # Split the lines into training and validation sets
    split_index = int(len(lines) * (1 - valid_ratio))
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]
    
    # Save the training and validation sets
    with open(train_file, "w", encoding="utf-8") as train_out:
        train_out.writelines(train_lines)
    
    with open(valid_file, "w", encoding="utf-8") as valid_out:
        valid_out.writelines(valid_lines)
    
    print(f"Split completed! Training set: {len(train_lines)} lines, Validation set: {len(valid_lines)} lines.")

# Example usage
split_txt_file(
    input_file="ic15_rec\\rec_gt_train.txt",  # Replace with your input file name
    train_file="ic15_rec\\rec_gt_train.txt",
    valid_file="ic15_rec\\rec_gt_valid.txt",
    valid_ratio=0.1  # 10% validation
)
