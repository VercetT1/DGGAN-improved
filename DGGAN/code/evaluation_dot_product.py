import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from scipy.spatial.distance import cdist
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinkPrediction():
    def __init__(self, config):
        self.links = [[], [], []]
        sufs = ['_0', '_50', '_100']
        for i, suf in enumerate(sufs):
            with open(config.test_file + suf) as infile:
                for line in infile.readlines():
                    s, t, label = [int(item) for item in line.strip().split()]
                    self.links[i].append([s, t, label])

    def evaluate(self, embedding_matrix):
      test_y = [[], [], []]
      pred_y = [[], [], []]
      pred_label = [[], [], []]
      node_ids = [[], [], []]  # Added to store node IDs
      threshold = 0
      dot=[]
      train_links=[]
      '''
        n_node=55493
        number_of_nodes=10000
        emd_dim=128
        selected_nodes = random.sample(range(n_node), number_of_nodes)  # Randomly select nodes
        selected_nodes = sorted(selected_nodes)
        seleceted_nodes=set(selected_nodes)
        selected_embeddings = embedding_matrix[:, selected_nodes, :]
        distances = []
        for i in range(len(selected_embeddings[0])):
            emb1 = selected_embeddings[0][random.randrange(number_of_nodes)]
            emb2 = selected_embeddings[1][random.randrange(number_of_nodes)]
            distance = np.linalg.norm(emb1 - emb2)
            distances.append(1/(1+distance))
        distances = np.array(distances)
        #flattened_matrix = np.reshape(embedding_matrix[:, selected_nodes, :], (2 * number_of_nodes, embedding_matrix.shape[2]))
        #distances = cdist(flattened_matrix, flattened_matrix, metric='euclidean')
        #distances = cdist(embedding_matrix[:, selected_nodes, :], embedding_matrix[:, selected_nodes, :], metric='euclidean')
        sorted_distances=-np.sort(-distances)
        #sorted_distances=set(sorted_distances)
        #sorted_distances = np.array(list(sorted_distances))
        percentile_5 = np.percentile(sorted_distances, 5)
        threshold=percentile_5
        print('Sorted distances length: '+str(len(sorted_distances)))
        print('\n' + 'sorted distances: ')
        print(sorted_distances)
        '''
      '''
        pred_y_temp= [[], [], []]
        for i in range(len(self.links)):
            for s, t, label in self.links[i]:
                distance_temp = np.linalg.norm(embedding_matrix[0][s] - embedding_matrix[1][t])
                pred_y_temp[i].append(distance_temp)
        combined_list = [item for sublist in pred_y_temp for item in sublist]
        combined_list.sort()
        percentile_5 = np.percentile(combined_list, 5) 
        threshold=percentile_5
        print(f"The 5th percentile value is: {threshold}")
        '''
      '''
        for i in range(len(self.links)):
            for s, t, label in self.links[i]:
                test_y[i].append(label)
                distance = (1/(1+(np.linalg.norm(embedding_matrix[0][s] - embedding_matrix[1][t]))))
                #pred_y[i].append(distance)
                pred_y[i].append(distance)
                if distance >= threshold:
                    pred_label[i].append(1)
                else:
                    pred_label[i].append(0)
                node_ids[i].append((s, t))  # Store node IDs
         '''
      with open('/content/drive/MyDrive/DGGAN/data/cite/train_0.5.txt') as infile:
                for line in infile.readlines():
                    source, target = [int(item) for item in line.strip().split()]
                    train_links.append([source, target])
      for s, t in train_links:
          dot.append((embedding_matrix[0][s].dot(embedding_matrix[1][t])))       
      dot.sort()
      percentile_5 = np.percentile(dot, 5) 
      threshold=percentile_5 
      for i in range(len(self.links)):
          for s, t, label in self.links[i]:
              test_y[i].append(label)
              pred_y[i].append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
              if pred_y[i][-1] >= threshold:
                  pred_label[i].append(1)
              else:
                  pred_label[i].append(0) 
              node_ids[i].append((s, t))            
      auc = [0, 0, 0]
      true_predictions=0
      false_predictions=0
      with open('total_result.txt', 'w') as outfile:
        for i in range(len(test_y)):
          auc[i] = roc_auc_score(test_y[i], pred_label[i])
          fpr, tpr, _ = roc_curve(test_y[i], pred_label[i])
          plt.plot(fpr, tpr, 'b-', label='ROC Curve')
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('Receiver Operating Characteristic (ROC) Curve')
          plt.legend()
          plt.show()
              #auc[i] = roc_auc_score(test_y[i], pred_label[i])
              #auc[i] = roc_auc_score(test_y[i], pred_label[i])  
              #for i in range(len(test_y)):
          outfile.write(f"AUC-{i}: {auc[i]:.4f}\n")
          for j in range(len(test_y[i])):
            outfile.write("Node IDs: " + ' '.join(map(str, node_ids[i][j])) + '\n')  # Write node IDs
            outfile.write("Test Label: " + str(test_y[i][j]) + '\n')
            outfile.write("Predicted Value: " + str(pred_y[i][j]) + '\n')
            if(test_y[i][j] == pred_label[i][j]):
                true_predictions=true_predictions+1
            else:
              false_predictions=false_predictions+1
              '''
            if(test_y[i][j] == 1 and pred_label[i][j] ==1):
              true_positive=true_positive+1
            elif(test_y[i][j] == 1 and pred_label[i][j] ==0):
              false_positive=false_positive+1
            elif(test_y[i][j] == 0 and pred_label[i][j] ==0):
              true_negative=true_negative+1
            elif(test_y[i][j] == 0 and pred_label[i][j] ==1):
              false_negative=false_negative+1  
              '''            
            outfile.write("Predicted Label: " + str(pred_label[i][j]) + '\n')
            outfile.write('\n') 
        outfile.write('\n')
        outfile.write('True predictions: '+str(true_predictions))
        outfile.write('\n')
        outfile.write('False predictions: '+str(false_predictions))
        outfile.write('\n')
        outfile.write('Thershold: '+str(threshold))
        outfile.write('\n')
        return auc
