
import json 
from collections import Counter 
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image


def get_questions(questions_path = 'dataset/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json'): 
    
    with open(questions_path, 'r') as file:
        questions_file = json.load(file)
    questions_collection = questions_file["questions"]
    questions_list = [question["question"] for question in questions_collection]

    return questions_collection,questions_list    
            

def get_answers(answers_path = 'dataset/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json'): 
    
    with open(answers_path, 'r') as file:
        answers_file = json.load(file)
    
    answers_list = answers_file["annotations"]

    answers = [answer["multiple_choice_answer"] for answer in answers_list]

    return answers    


def get_images(questions_dataset, split= "train"): 
    
    images_list= list()

    # Randomly select a data entry
    for data_entry in questions_dataset: 

        # Format the image ID with leading zeros
        image_id_formatted = str(data_entry['image_id']).zfill(12)
        image_path = f"dataset/{split}2014/COCO_{split}2014_{image_id_formatted}.jpg"
        images_list.append(image_path)

    return images_list


def create_label_mapping(answers_dataset, top_n= 1000):
    
    # Analyze answers

    answers_counter = Counter()

    for answerlet in answers_dataset: 
        answers_counter.update([answerlet])
    
    most_common_answers = [ans for ans, _ in answers_counter.most_common(top_n)]

    # Create a label mapping
    answer_to_label = {ans: i for i, ans in enumerate(most_common_answers)}

    # Convert answers to labels
    labels = [answer_to_label.get(ans, -1) for ans in answers_counter]

    return answer_to_label, labels 
 

    

class VQAData(Dataset):
    """VQA dataset class for loading images, questions, and answers"""

    def __init__(self, questions_path, answers_path, split, transform=None):
        """
        Args:
            questions_path (str): Path to the questions dataset.
            answers_path (str): Path to the answers dataset.
            split (str): Dataset split ('train', 'val', 'test').
            transform (callable, optional): Optional transformations for image preprocessing.
        """
        self.questions_collection, self.questions_list = get_questions(questions_path)
        self.answers_collection = get_answers(answers_path)
        self.images_collection = get_images(self.questions_collection, split=split)

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match CNN input size
            transforms.ToTensor(),         # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
        ])

    def __len__(self):
        return len(self.images_collection)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        question = self.questions_list[idx]
        answer = self.answers_collection[idx]
        img_path = self.images_collection[idx]

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return question, answer, image


def get_dataloaders(): 

    train_questions_path = 'dataset/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json'
    train_answers_path = 'dataset/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json'
    train_set = VQAData(train_questions_path, train_answers_path, "train")


    val_questions_path = 'dataset/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json'
    val_answers_path = 'dataset/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json'
    val_set = VQAData(val_questions_path, val_answers_path, "val")


    train_loader = DataLoader(train_set, batch_size=8, shuffle= True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle= False)
    return train_loader, val_loader



