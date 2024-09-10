from torch.utils.data import Dataset

class gen_partial_dataset(Dataset):
    def __init__(self, images, given_label_matrix):
        self.images = images
        self.given_label_matrix = given_label_matrix
        
    def __len__(self):
        return len(self.given_label_matrix)
        
    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        
        return each_image, each_label
