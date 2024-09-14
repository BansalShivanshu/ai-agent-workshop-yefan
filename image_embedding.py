import math
from sentence_transformers import SentenceTransformer, util
import os.path
import pickle
from scipy import spatial
from PIL import Image
import faiss
import numpy as np

image_pkl_path = 'clip_embedding.pkl'

# Add description, and tutorial type of comments
class image_embedding_store:
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.embedding_dict = self.update_imgage_embedding()

        self.id_to_name = {}
        self.embedding_list = []
        next_id = 0
        for file_name, embedding in self.embedding_dict.items():
            self.id_to_name[next_id] = file_name
            self.embedding_list.append(embedding)
            next_id = next_id + 1

    def update_imgage_embedding(self):
        embedding_dict = {}

        if os.path.isfile(image_pkl_path):
            with open(image_pkl_path, 'rb') as file:
                embedding_dict = pickle.load(file)

        embedding_dict_updated = False
        file_in_dataset = []
        for file in os.listdir(self.dataset_dir):
            filename = os.fsdecode(file)
            file_in_dataset.append(filename)
            if filename in embedding_dict:
                continue
            embedding_dict_updated = True
            print(f"Generating embedding for {filename}")

            embedding_dict[filename] = self.model.encode(Image.open(f"{self.dataset_dir}/{filename}"))

        all_files = list(embedding_dict.keys())
        files_not_in_dataset = [x for x in all_files if x not in file_in_dataset]
        for file in files_not_in_dataset:
            del embedding_dict[file]
            embedding_dict_updated = True

        print(f"Number of images in dataset: {len(embedding_dict)}")

        if embedding_dict_updated:
            with open(image_pkl_path, "wb") as file:
                pickle.dump(embedding_dict, file)

        return embedding_dict

    def get_all_files(self):
        return list(self.embedding_dict.keys())
    
    def find_closest_image_by_linear_search(self, description):
        text_embedding = self.model.encode(description)
        max_similarity = -1
        max_item_name = ""
        for name, vec in self.embedding_dict.items():
            # EXPERIMENT: Try different similarity metrics from the util module
            similarity = util.cos_sim(text_embedding, vec)
            if similarity > max_similarity:
                max_similarity = similarity
                max_item_name = name
        return max_item_name

    def find_top_k_similar_images(self, description, k=3):
        text_embedding = self.model.encode(description)
        distances = []
        for name, vec in self.embedding_dict.items():
            similarity = util.cos_sim(text_embedding, vec)
            distances.append((name, similarity))

        # Sort by similarity in descending order
        distances.sort(key=lambda x: x[1], reverse=True)

        # Get the top k similar images
        top_k_images = [name for name, _ in distances[:k]]
        return top_k_images

    def find_mmr_images(self, description, k=3):
        text_embedding = self.model.encode(description)
        distances = []
        for name, vec in self.embedding_dict.items():
            similarity = util.cos_sim(text_embedding, vec)
            distances.append((name, similarity))

        distances.sort(key=lambda x: x[1], reverse=True)
        top_k_images = []
        top_k_similarities = []
        tolerance = 0.01
        for dist in distances:
            if not top_k_images:
                top_k_images.append(dist[0])
                top_k_similarities.append(dist[1])
            else:
                if math.fabs(dist[1] - top_k_similarities[-1]) > tolerance:
                    top_k_images.append(dist[0])
                    top_k_similarities.append(dist[1])
            if len(top_k_images) == k:
                break

        return top_k_images

    # Vector search algorithms
    def find_top_k_by_kd_tree(self, description, k=3):
        # Question: What's the time complexity of building the kd tree? and what's the complexity of querying the kd tree?
        tree = spatial.KDTree(self.embedding_list)

        text_embedding = self.model.encode(description)
        dist, index = tree.query(text_embedding, k=k)

        if k==1:
            return [self.id_to_name[index]]
        return [self.id_to_name[i] for i in index]
    
    def find_top_k_by_faiss(self, description, k=3):
        text_embedding = np.array([self.model.encode(description)]).astype('float32')
        embedding_array = np.array(self.embedding_list).astype('float32')

        # EXPERIMENT: Try a different type of index. What's its pros and cons.
        index = faiss.IndexFlatL2(embedding_array.shape[1]) 
        index.add(embedding_array)

        distances, indices = index.search(text_embedding, k)

        top_k_images = [self.id_to_name[i] for i in indices[0]]
        return top_k_images

# Test stub for the different search algorithms
if __name__ == "__main__":
    dataset_dir = 'dataset'
    # Example usage of the image embedding functionalities
    img_store = image_embedding_store(dataset_dir)  # Assuming ImageEmbedding is the class name

    query = "a cat reading a book"

    closest_image = img_store.find_closest_image_by_linear_search(query)
    print(f"Closest image: {closest_image}")

    top_k_images = img_store.find_top_k_by_kd_tree(query)
    print(f"Top k images: {top_k_images}")

    top_k_images = img_store.find_top_k_by_faiss(query)
    print(f"Top k images: {top_k_images}")

    top_k_mmr = img_store.find_mmr_images(query)
    print(f"Top k mmr images: {top_k_mmr}")
