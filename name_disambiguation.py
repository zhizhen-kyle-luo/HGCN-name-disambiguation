import pickle
import os
import re
import numpy as np
import networkx as nx
from gensim.models import word2vec
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import xml.dom.minidom
import xml.etree.ElementTree as ET
from GCN import *
import community
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import *
import csv
from scipy.sparse.csgraph import connected_components
import random
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import copy
import json
import sys
import argparse # Add argparse import

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow as tf2 #Tensorflow 2.x

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
        

def GHAC(mlist,G,idx_pid,n_clusters=-1):

    distance=[]
    graph=[]
    for i in range(len(mlist)):
        gtmp=[]
        for j in range(len(mlist)):
            if i<j and G.has_edge(idx_pid[i],idx_pid[j]):
                cosdis=1/(1+np.exp(-np.dot(mlist[i],mlist[j])))
                gtmp.append(cosdis)
            elif i>j:
                gtmp.append(graph[j][i])
            else:
                gtmp.append(0)
        graph.append(gtmp)

    graph=np.array(graph)
    distance =np.multiply(graph,-1)

    if n_clusters==-1:
        best_m=-10000000

        n_components, labels = connected_components(graph)
        Gr=nx.from_numpy_matrix(graph)

        graph[graph<=0.9]=0 #Edge pre-clustering
        n_components1, labels = connected_components(graph)

        for k in range(n_components1,n_components-1,-1):
            model_HAC = AgglomerativeClustering(linkage="average",metric='precomputed',n_clusters=k)
            model_HAC.fit(distance)
            labels = model_HAC.labels_

            part= {}
            for j in range (len(labels)):
                part[j]=labels[j]

            mod = community.modularity(part,Gr)
            if mod>=best_m:
                best_m=mod
                best_labels=labels
        labels = best_labels
    else:
        model_HAC = AgglomerativeClustering(linkage="average",metric='precomputed',n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_

    return labels


def pairwise_evaluate(correct_labels,pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1


def generate_author_id_clusters(fname, correct_labels, predicted_labels):
    """
    Creates and saves a mapping from person clusters to author IDs.
    
    For a given name (like "Li Shen"), this shows which person cluster
    corresponds to which OpenAlex IDs in the dataset.
    
    Args:
        fname: Name being disambiguated
        correct_labels: List of true author IDs for each paper
        predicted_labels: List of predicted cluster labels for each paper
    """
    # Load the mapping from numeric labels to actual OpenAlex IDs
    author_id_mapping = {}
    
    # Open the XML file to get the ID mapping
    xml_path = os.path.join("raw-data-temp", f"{fname}.xml")
    try:
        # Parse the XML file to get the author ID mapping
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Clean the XML content
        xml_content = re.sub(u"&", u" ", xml_content)
        root = ET.fromstring(xml_content)
        
        # Get the actual OpenAlex ID from personID element
        person_id = root.find('personID').text
        
        # Get all the author IDs and their labels from publications
        for pub in root.findall('publication'):
            # The numeric label used in correct_labels
            label = pub.find('label').text
            if label not in author_id_mapping:
                # Load the ID from the cache file
                cache_path = os.path.join("cache", f"{fname}_data.json")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Get the reverse mapping from label to author ID
                    label_to_author = {v: k for k, v in cache_data["author_id_to_label"].items()}
                    # For each label, get the corresponding author ID
                    for numeric_label, author_id in label_to_author.items():
                        author_id_mapping[numeric_label] = author_id
                else:
                    # Fallback to using the first personID from XML if cache not available
                    author_id_mapping[label] = person_id
    except Exception as e:
        print(f"Error loading ID mapping: {e}")
        # Use the labels as-is if we can't get the real IDs
        for label in set(correct_labels):
            author_id_mapping[str(label)] = f"A{label}"
    
    # Track which OpenAlex IDs have been assigned to which clusters
    assigned_ids = set()
    
    # First, map papers to their predicted clusters and true author IDs
    paper_clusters = {}
    for i in range(len(predicted_labels)):
        pred_cluster = int(predicted_labels[i])
        true_author_label = str(int(correct_labels[i]))
        true_author_id = author_id_mapping.get(true_author_label, true_author_label)
        
        if pred_cluster not in paper_clusters:
            paper_clusters[pred_cluster] = []
        
        paper_clusters[pred_cluster].append((i, true_author_id))
    
    # Now analyze the clusters to create proper person clusters
    # Determine the majority author ID in each predicted cluster
    cluster_to_author_ids = {}
    for cluster, papers in paper_clusters.items():
        # Count occurrences of each ID
        id_counts = {}
        for _, author_id in papers:
            id_counts[author_id] = id_counts.get(author_id, 0) + 1
        
        # Create a sorted list of (count, id) pairs
        sorted_ids = sorted([(count, id) for id, count in id_counts.items()], reverse=True)
        
        # Assign IDs to this cluster if they haven't been assigned elsewhere
        # or if this cluster has more occurrences of the ID
        cluster_author_ids = set()
        for count, author_id in sorted_ids:
            if author_id not in assigned_ids:
                cluster_author_ids.add(author_id)
                assigned_ids.add(author_id)
        
        if cluster_author_ids:
            cluster_to_author_ids[cluster] = sorted(list(cluster_author_ids))
    
    # Create a new dictionary with sequential indices as keys
    person_clusters = {}
    for idx, (cluster, author_ids) in enumerate(cluster_to_author_ids.items()):
        # Convert all IDs to strings for consistent formatting
        person_clusters[str(idx)] = [str(id) for id in author_ids]
    
    # Save the result to a JSON file
    # Create directory if it doesn't exist
    os.makedirs('result/author_clusters', exist_ok=True)
    
    with open(f'result/author_clusters/{fname}_clusters.json', 'w') as f:
        json.dump({fname: person_clusters}, f, indent=2)
    
    # Also print to console
    print(f"\n{fname} person clusters to author IDs mapping:")
    print(json.dumps({fname: person_clusters}, indent=2))
    
    return person_clusters

# Add this function to encapsulate original output logic
def print_original_output(fname, correct_labels, labels, pairwise_precision, pairwise_recall, pairwise_f1):
    """Print the original output format (can be enabled/disabled as needed)"""
    print(correct_labels, len(set(correct_labels)))
    print(list(labels), len(set(list(labels))))
    print(fname, pairwise_precision, pairwise_recall, pairwise_f1)

def disambiguate_openAlex_ids(fname):
    """
    Process OpenAlex author IDs for a given name and disambiguate them.
    
    This function is used for testing the model on OpenAlex data without ground truth.
    It loads data prepared by openAlex_to_HGCN.py and performs clustering to
    determine which OpenAlex author IDs likely represent the same person.
    
    Args:
        fname: Author name whose IDs need to be disambiguated
    
    Returns:
        Dictionary mapping predicted person clusters to sets of OpenAlex IDs
    """
    print(f"\nDisambiguating OpenAlex IDs for author: {fname}")
    
    # Define the regex pattern for cleaning text
    r = '[!""#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'
    
    # Define stopwords
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','method','algrithom','by','model']
    stopword = [porter_stemmer.stem(w) for w in stopword]
    
    # Load cached data which contains the mapping of author IDs
    cache_path = os.path.join("cache", f"{fname}_data.json")
    author_ids = []
    author_id_mapping = {}
    all_author_ids = []
    
    # Load the pre-trained word2vec model
    save_model_name = os.path.join(os.path.dirname(__file__), "gene", "word2vec.model")
    try:
        model_w = word2vec.Word2Vec.load(save_model_name)
        print(f"Loaded word2vec model from {save_model_name}")
    except Exception as e:
        print(f"Error loading word2vec model: {e}")
        print("Using random word vectors instead")
        
        # Create a dummy model with random vectors for words
        class DummyModel:
            def __init__(self):
                self.wv = {}
                
            def __contains__(self, word):
                return False
                
        model_w = DummyModel()
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Get all author IDs including those that might not have publications
        all_author_ids = list(cache_data["author_data"].keys())
        print(f"Found {len(all_author_ids)} total author IDs: {all_author_ids}")
        
        # Get the mapping used in the XML file
        author_id_mapping = cache_data["author_id_to_label"]
        author_ids = list(author_id_mapping.keys())
        print(f"Found {len(author_ids)} author IDs with publications: {author_ids}")
    else:
        print(f"No cached data found for {fname}. Please run openAlex_to_HGCN.py first.")
        return {}

    # Load the XML file to get publication data
    xml_path = os.path.join("raw-data-temp", f"{fname}.xml")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Clean the XML content
        xml_content = re.sub(u"&", u" ", xml_content)
        root = ET.fromstring(xml_content)
    except Exception as e:
        print(f"Error loading XML file: {e}")
        return {}

    # Create a mapping from publication index to author ID
    publication_to_author_id = {}
    for i, pub in enumerate(root.findall('publication')):
        label = pub.find('label').text
        for author_id, label_value in author_id_mapping.items():
            if label_value == label:
                publication_to_author_id[i] = author_id
                break
    
    # Process the data as in the main function
    p_to = {}
    p_t = {}
    
    # Process publications
    for i in root.findall('publication'):
        pid = i.find('id').text
        
        if pid in p_t:
            pid = pid + '1'
        
        line = i.find('title').text
        line = re.sub(r, ' ', line)
        line = line.replace('\t', ' ')
        line = line.lower()
        split_cut = line.split(' ')
        
        p_t[pid] = []
        p_to[pid] = []
        
        for j in split_cut:
            if len(j) > 1:
                p_to[pid].append(j)
                if porter_stemmer.stem(j) not in stopword:
                    p_t[pid].append(porter_stemmer.stem(j))
    
    # Construct PHNet
    pid_idx = {}
    idx_pid = {}
    idx = 0
    
    G = nx.Graph()
    
    for pid in p_t:
        G.add_node(pid)
        pid_idx[pid] = idx
        idx_pid[idx] = pid
        idx = idx + 1
    
    # Load co-author data
    Ga = nx.Graph()
    for pid in p_t:
        Ga.add_node(pid)
    
    fa = open("experimental-results/authors/" + fname + "_authorlist.txt", 'r', encoding='utf-8').readlines()
    
    for line in fa:
        line.strip()
        split_cut = line.split('\t')
        
        keyi = idx_pid[int(split_cut[0].strip())]
        keyj = idx_pid[int(split_cut[1].strip())]
        
        weights = 1
        if Ga.has_edge(keyi, keyj):
            Ga[keyi][keyj]['weight'] = Ga[keyi][keyj]['weight'] + weights
        else:
            Ga.add_edge(keyi, keyj, weight=weights)
    
    # Load co-venue data
    Gv = nx.Graph()
    for pid in p_t:
        Gv.add_node(pid)
    
    fv = open("experimental-results/" + fname + "_jconfpair.txt", 'r', encoding='utf-8').readlines()
    
    for line in fv:
        line.strip()
        split_cut = line.split('\t')
        
        keyi = idx_pid[int(split_cut[0].strip())]
        keyj = idx_pid[int(split_cut[1].strip())]
        weights = 1
        Gv.add_edge(keyi, keyj, weight=weights)
    
    # Construct co-title graph
    Gt = nx.Graph()
    for pid in p_t:
        Gt.add_node(pid)
    
    for i, keyi in enumerate(p_t):
        for j, keyj in enumerate(p_t):
            weights = len(set(p_t[keyi]).intersection(set(p_t[keyj])))
            if j > i and weights >= 2:
                Gt.add_edge(keyi, keyj, weight=weights)
    
    # Combine graphs
    Glist = [Ga, Gt, Gv]
    
    for i in range(len(Glist)):
        for u, v, d in Glist[i].edges(data='weight'):
            if G.has_edge(u, v):
                G[u][v]['weight'] += d
            else:
                G.add_edge(u, v, weight=d)
    
    Glist.append(G)
    
    # Sampling paths
    all_neighbor_samplings = []
    all_neg_sampling = []
    
    for i, Gi in enumerate(Glist):
        adj_matrix = nx.to_numpy_array(Gi)
        Gtmp = copy.deepcopy(Gi)
        for u, v, d in Gtmp.edges(data='weight'):
            Gtmp[u][v]['weight'] = 1
        length = dict(nx.all_pairs_dijkstra_path_length(Gtmp))
        
        for u in length:
            for v in length[u]:
                if Gtmp.has_edge(u, v) is False and length[u][v] > 0:
                    Gtmp.add_edge(u, v, weight=length[u][v])
        pathl_matrix = nx.to_numpy_array(Gtmp)
        
        neighbor_samplings = []
        neg_samplings = []
        for i in range(G.number_of_nodes()):
            node_weights = adj_matrix[i]
            if np.sum(node_weights) == 0:
                neighbor_samplings.append(0)
            else:
                weight_distribution = node_weights / np.sum(node_weights)
                neighbor_samplings.append(AliasSampling(weight_distribution))
            
            node_i_degrees = pathl_matrix[i]
            node_i_degrees[node_i_degrees == 0] = 6
            node_i_degrees[i] = 0
            node_i_degrees[node_i_degrees <= 1] = 0
            
            if np.sum(node_i_degrees) == 0:
                neg_samplings.append(0)
            else:
                node_distribution = node_i_degrees / np.sum(node_i_degrees)
                neg_samplings.append(AliasSampling(node_distribution))
        
        all_neighbor_samplings.append(neighbor_samplings)
        all_neg_sampling.append(neg_samplings)
    
    # Random walks
    numwalks = 4
    walklength = 10
    negative_num = 3
    
    u_i = []
    u_j = []
    label = []
    metapath = [0, 1, 0, 2]
    
    for node_index in range(G.number_of_nodes()):
        for j in range(0, numwalks):
            node_start = node_index
            g_index = j
            gi = metapath[g_index]
            for i in range(0, walklength):
                if all_neighbor_samplings[gi][node_start] != 0:
                    node_p = all_neighbor_samplings[gi][node_start].sampling()
                    u_i.append(node_start)
                    u_j.append(node_p)
                    label.append(1)
                    
                    if all_neg_sampling[-1][node_start] != 0:
                        for k in range(negative_num):
                            node_n = all_neg_sampling[-1][node_start].sampling()
                            u_i.append(node_start)
                            u_j.append(node_n)
                            label.append(-1)
                    
                    g_index = (g_index + 1) % len(metapath)
                    gi = metapath[g_index]
                    
                    if all_neighbor_samplings[gi][node_p] != 0:
                        node_p1 = all_neighbor_samplings[gi][node_p].sampling()
                        u_i.append(node_start)
                        u_j.append(node_p1)
                        label.append(1)
                        
                        if all_neg_sampling[-1][node_start] != 0:
                            for k in range(negative_num):
                                node_n = all_neg_sampling[-1][node_start].sampling()
                                u_i.append(node_start)
                                u_j.append(node_n)
                                label.append(-1)
                    
                    node_start = node_p
                else:
                    for k in range(negative_num):
                        node_n = all_neg_sampling[-1][node_start].sampling()
                        u_i.append(node_start)
                        u_j.append(node_n)
                        label.append(-1)
                    g_index = (g_index + 1) % len(metapath)
                    gi = metapath[g_index]
    
    # Generate node attributes
    node_attr = []
    for pid in p_to:
        words_vec = []
        for word in p_to[pid]:
            if word in model_w.wv:
                words_vec.append(model_w.wv[word])
        if len(words_vec) == 0:
            words_vec.append(2 * np.random.random(100) - 1)
        node_attr.append(np.mean(words_vec, 0))
    node_attr = np.array(node_attr)
    
    # Train model
    batch_size = 64
    total_batch = 3 * int(len(u_i) / batch_size)
    display_batch = 100
    
    model = GCN(Glist, node_attr, batch_size=batch_size)
    
    avg_loss = 0.
    for i in range(total_batch):
        sdx = (i * batch_size) % len(u_i)
        edx = ((i + 1) * batch_size) % len(u_i)
        
        if edx > sdx:
            u_ii = u_i[sdx:edx]
            u_jj = u_j[sdx:edx]
            labeli = label[sdx:edx]
        else:
            u_ii = u_i[sdx:] + u_i[0:edx]
            u_jj = u_j[sdx:] + u_j[0:edx]
            labeli = label[sdx:] + label[0:edx]
        
        loss = model.train_line(u_ii, u_jj, labeli)
        avg_loss += loss / display_batch
        
        if i % display_batch == 0 and i > 0:
            print('%d/%d loss %8.6f' % (i, total_batch, avg_loss))
            avg_loss = 0.
    
    # Calculate embeddings and cluster
    embed_matrix = model.cal_embed()
    
    # --- Modularity-based cluster number determination ---
    # Build similarity graph from embeddings
    similarity_graph_matrix = np.zeros((len(embed_matrix), len(embed_matrix)))
    for i in range(len(embed_matrix)):
        for j in range(i + 1, len(embed_matrix)):
            # Check if an edge exists in the combined graph Glist[-1] before calculating similarity
            # Note: This assumes idx_pid maps indices 0..N-1 to publication IDs
            node_i_pid = idx_pid.get(i)
            node_j_pid = idx_pid.get(j)
            if node_i_pid and node_j_pid and Glist[-1].has_edge(node_i_pid, node_j_pid):
                 # Using cosine similarity directly on embeddings might be better than sigmoid
                 # similarity = cosine_similarity(embed_matrix[i].reshape(1,-1), embed_matrix[j].reshape(1,-1))[0][0]
                 # Replicating original GHAC similarity for consistency:
                 similarity = 1 / (1 + np.exp(-np.dot(embed_matrix[i], embed_matrix[j])))
                 similarity_graph_matrix[i, j] = similarity
                 similarity_graph_matrix[j, i] = similarity # Ensure symmetry
                 
    # Use the combined graph Glist[-1] for modularity calculation
    Gr = Glist[-1] 
    
    # Calculate connected components on the similarity graph
    # Use a threshold similar to original GHAC? (e.g., 0.9)
    threshold = 0.9 
    thresholded_graph_matrix = similarity_graph_matrix.copy()
    thresholded_graph_matrix[thresholded_graph_matrix <= threshold] = 0
    
    # Get components from thresholded and original similarity graphs
    n_components1, _ = connected_components(thresholded_graph_matrix, directed=False, connection='weak')
    n_components, _ = connected_components(similarity_graph_matrix, directed=False, connection='weak')
    
    best_m = -1.0 # Modularity is between -0.5 and 1
    best_labels = None
    best_k = -1

    print(f"Searching for optimal clusters between {n_components1} and {n_components} based on modularity...")
    
    # Calculate distance matrix once (negative similarity)
    distance_matrix = np.multiply(similarity_graph_matrix, -1)
    
    # Iterate through possible cluster numbers
    # Ensure range is valid (start <= end)
    start_k = max(2, n_components1) # Avoid k=1, ensure at least 2 clusters if possible
    end_k = max(start_k, n_components) # Ensure end is not less than start

    # If only one component overall, default to a reasonable number or len(author_ids)
    if end_k <= 1:
         print("Graph has only one major component. Defaulting k based on author IDs.")
         # Fallback to previous heuristic or len(author_ids) if only 1 component
         best_k = max(2, min(len(author_ids), 10)) 
         if best_k <= 1 and len(embed_matrix) > 1: # Ensure k>=2 if more than 1 item
             best_k = 2
         elif len(embed_matrix) <=1: # Handle edge case of 0 or 1 publication
             best_k = 1
    else:
        for k in range(start_k, end_k + 1):
            # Cluster using Agglomerative Clustering
            model_HAC = AgglomerativeClustering(linkage="average", metric='precomputed', n_clusters=k)
            
            # Fit HAC model
            # Add small epsilon for stability if needed, though precomputed shouldn't require it
            current_labels = model_HAC.fit_predict(distance_matrix) 
            
            # Prepare partition for modularity calculation
            part = {}
            label_map = {}
            current_label_index = 0
            # Map networkx node IDs (pids) to cluster labels
            for node_idx, cluster_label in enumerate(current_labels):
                pid = idx_pid.get(node_idx)
                if pid and pid in Gr: # Ensure the pid exists in the modularity graph Gr
                     part[pid] = cluster_label
                     
            # Calculate modularity using the combined graph Gr
            if len(part) > 0 and len(Gr) > 0 and len(Gr.edges()) > 0:
                 # Ensure the partition covers nodes present in the graph Gr
                 valid_partition = {node: label for node, label in part.items() if node in Gr}
                 if len(valid_partition) > 1 : # Modularity requires graph and partition > 1 node
                      mod = community.modularity(valid_partition, Gr, weight='weight')
                      #print(f"k={k}, Modularity={mod}") # Debug print
                      if mod > best_m:
                           best_m = mod
                           best_labels = current_labels
                           best_k = k
                 else:
                      #print(f"k={k}, Partition too small for modularity calculation.")
                      if best_k == -1: # Handle cases where only k=2 might be valid but partition small
                           best_labels = current_labels
                           best_k = k
            else:
                 #print(f"k={k}, Graph or partition empty, skipping modularity.")
                 # Handle case where no edges or nodes lead to fallback
                 if best_k == -1: 
                      best_labels = current_labels
                      best_k = k

        # If no suitable k found (e.g., modularity calculation failed), fallback
        if best_k == -1 or best_labels is None:
             print("Could not determine optimal k via modularity. Falling back.")
             # Fallback to a reasonable default or heuristic
             best_k = max(2, min(len(author_ids), 10)) 
             if best_k <= 1 and len(embed_matrix) > 1:
                 best_k = 2
             elif len(embed_matrix) <=1:
                 best_k = 1
             # Rerun clustering with fallback k
             model_HAC = AgglomerativeClustering(linkage="average", metric='precomputed', n_clusters=best_k)
             best_labels = model_HAC.fit_predict(distance_matrix)

    print(f"Optimal number of clusters determined: k={best_k} (Modularity: {best_m:.4f})")
    labels = best_labels # Use the labels found for the best k
    # --- End of modularity logic ---

    # Create mapping from publication index to cluster
    pub_clusters = {}
    for i, cluster_id in enumerate(labels):
        # Ensure mapping uses the determined best_labels
        actual_cluster_id = int(cluster_id) # Convert numpy int potentially
        if i < len(publication_to_author_id): # Check bounds
            author_id = publication_to_author_id.get(i)
            if author_id:
                if actual_cluster_id not in pub_clusters:
                    pub_clusters[actual_cluster_id] = []
                pub_clusters[actual_cluster_id].append(author_id)
    
    # Create final mapping from clusters to author IDs
    # Ensure each author ID only appears in one cluster
    assigned_ids = set()
    cluster_to_author_ids = {}
    
    for cluster, author_ids in pub_clusters.items():
        # Count occurrences of each ID in this cluster
        id_counts = {}
        for author_id in author_ids:
            id_counts[author_id] = id_counts.get(author_id, 0) + 1
        
        # Sort by frequency
        sorted_ids = sorted([(count, id) for id, count in id_counts.items()], reverse=True)
        
        # Assign IDs that haven't been assigned elsewhere
        cluster_author_ids = []
        for _, author_id in sorted_ids:
            if author_id not in assigned_ids:
                cluster_author_ids.append(author_id)
                assigned_ids.add(author_id)
        
        if cluster_author_ids:
            cluster_to_author_ids[cluster] = cluster_author_ids
    
    # Handle IDs without publications (not in clusters yet)
    unassigned_ids = set(all_author_ids) - assigned_ids
    
    if unassigned_ids:
        print(f"Found {len(unassigned_ids)} author IDs without publications: {unassigned_ids}")
        # Create a new cluster for each unassigned ID
        next_cluster_id = max(cluster_to_author_ids.keys()) + 1 if cluster_to_author_ids else 0
        for unassigned_id in unassigned_ids:
            cluster_to_author_ids[next_cluster_id] = [unassigned_id]
            next_cluster_id += 1
    
    # Create the final output format
    person_clusters = {}
    for idx, (cluster, ids) in enumerate(cluster_to_author_ids.items()):
        person_clusters[str(idx)] = sorted(list(set(ids)))
    
    # Save result
    os.makedirs('result/author_clusters', exist_ok=True)
    with open(f'result/author_clusters/{fname}_openAlex_clusters.json', 'w') as f:
        json.dump({fname: person_clusters}, f, indent=2)
    
    # Print result (Similar to generate_author_id_clusters)
    print(f"\n{fname} OpenAlex author ID clusters:")
    print(json.dumps({fname: person_clusters}, indent=2))
    
    return person_clusters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGCN Name Disambiguation')
    parser.add_argument('--name', type=str, help='Author name to process (required for default OpenAlex mode).')
    parser.add_argument('--standard', action='store_true', help='Run in Standard mode using raw-data-temp/ XMLs and labels for evaluation.')
    
    args = parser.parse_args()

    # --- Standard Mode --- 
    if args.standard:
        print("--- Running in Standard Mode (Processing files from raw-data-temp/, using labels) ---")
        # Check if --name was also provided, if so, warn that it's ignored in standard mode
        if args.name:
             print("Warning: --name argument is ignored when running in --standard mode.")
             
        # Pre-trained word2vec model
        save_model_name = "gene/word2vec.model"
        model_w = word2vec.Word2Vec.load(save_model_name)

        # Regex and stopwords
        r = '[!""#$%&\'()*+,-./:;<=>?@[\]^_`{|}~—～]+'
        stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','method','algrithom','by','model']
        stopword = [porter_stemmer.stem(w) for w in stopword]

        result = [] # Stores [name, P, R, F1] for averaging
        path = "raw-data-temp/"
        
        if not os.path.exists(path) or not os.listdir(path):
             print(f"Error: Directory {path} is empty or does not exist. Standard mode needs XML files here.")
             sys.exit(1)
             
        file_names = os.listdir(path)
        
        # Process each file individually
        for fname_with_ext in file_names:
            if not fname_with_ext.endswith('.xml'):
                continue
                
            # Extract name without extension
            author_name = fname_with_ext[:-4]
            print(f"\nProcessing {author_name}...")
            
            # Initialize variables for this author
            p_t = {}
            p_to = {}
            correct_labels = []
            
            # Read and process XML file
            try:
                with open(os.path.join(path, fname_with_ext), 'r', encoding = 'utf-8') as f_xml:
                     xml_content = f_xml.read()
                text = re.sub(u"&", u" ", xml_content)
                root = ET.fromstring(text)
            except ET.ParseError as e:
                print(f"Error parsing XML file {fname_with_ext}: {e}")
                continue # Skip to next file
            except FileNotFoundError:
                 print(f"Error: Could not find file {os.path.join(path, fname_with_ext)}")
                 continue # Skip to next file


            # Process publications (Ensure file structure is correct)
            publications = root.findall('publication')
            if not publications:
                 print(f"Warning: No <publication> tags found in {fname_with_ext}. Skipping graph construction.")
                 continue # Skip graph construction if no publications
                 
            for i in publications:
                try:
                    pid_elem = i.find('id')
                    label_elem = i.find('label')
                    title_elem = i.find('title')

                    if pid_elem is None or label_elem is None or title_elem is None or pid_elem.text is None or label_elem.text is None or title_elem.text is None:
                         print(f"Warning: Skipping publication in {fname_with_ext} due to missing id, label, or title.")
                         continue

                    pid = pid_elem.text
                    
                    if pid in p_t:
                        pid = pid + '1'
                        
                    correct_labels.append(int(label_elem.text))
                    
                    line = title_elem.text
                    line = re.sub(r, ' ', line)
                    line = line.replace('\t', ' ')
                    line = line.lower()
                    split_cut = line.split(' ')
                    
                    p_t[pid] = []
                    p_to[pid] = []
                    
                    for j in split_cut:
                        if len(j) > 1:
                            p_to[pid].append(j)
                            if porter_stemmer.stem(j) not in stopword:
                                p_t[pid].append(porter_stemmer.stem(j))
                except ValueError:
                     print(f"Warning: Skipping publication in {fname_with_ext} due to invalid label format (not an integer).")
                     continue
                except Exception as e:
                     print(f"Warning: Unexpected error processing a publication in {fname_with_ext}: {e}")
                     continue
                     
            # --- Guard against empty data for graph construction ---
            if not p_t or not correct_labels:
                print(f"Warning: No valid publication data processed for {author_name}. Skipping model training and evaluation.")
                continue
            # --- End Guard ---
            
            # Construct PHNet
            pid_idx = {}
            idx_pid = {}
            idx = 0
            
            G = nx.Graph()
            
            for pid in p_t:
                G.add_node(pid)
                pid_idx[pid] = idx
                idx_pid[idx] = pid
                idx = idx + 1
            
            # Construct co-author graph
            Ga = nx.Graph()
            for pid in p_t:
                Ga.add_node(pid)
            
            author_file_path = f"experimental-results/authors/{author_name}_authorlist.txt"
            if os.path.exists(author_file_path):
                 try:
                    with open(author_file_path, 'r', encoding='utf-8') as fa_file:
                        fa_lines = fa_file.readlines()
                    
                    for line_num, line in enumerate(fa_lines):
                        line = line.strip()
                        if not line: continue # Skip empty lines
                        split_cut = line.split('\t')
                        if len(split_cut) < 2:
                             print(f"Warning: Skipping malformed line {line_num+1} in {author_file_path}")
                             continue
                        
                        try:
                            idx1 = int(split_cut[0].strip())
                            idx2 = int(split_cut[1].strip())
                            keyi = idx_pid.get(idx1) # Use .get for safety
                            keyj = idx_pid.get(idx2)
                            
                            if keyi is None or keyj is None:
                                 print(f"Warning: Invalid index found on line {line_num+1} in {author_file_path}. Skipping.")
                                 continue
                                 
                            weights = 1
                            if Ga.has_edge(keyi, keyj):
                                Ga[keyi][keyj]['weight'] = Ga[keyi][keyj]['weight'] + weights
                            else:
                                Ga.add_edge(keyi, keyj, weight=weights)
                        except ValueError:
                             print(f"Warning: Non-integer index found on line {line_num+1} in {author_file_path}. Skipping.")
                             continue
                 except FileNotFoundError:
                      print(f"Info: Author list file not found for {author_name}, skipping co-author graph edges.")
                 except Exception as e:
                      print(f"Error reading author list file {author_file_path}: {e}")
            else:
                 print(f"Info: Author list file not found for {author_name}, skipping co-author graph edges.")
            
            # Construct co-venue graph
            Gv = nx.Graph()
            for pid in p_t:
                Gv.add_node(pid)
            
            venue_file_path = f"experimental-results/{author_name}_jconfpair.txt"
            if os.path.exists(venue_file_path):
                 try:
                    with open(venue_file_path, 'r', encoding='utf-8') as fv_file:
                         fv_lines = fv_file.readlines()

                    for line_num, line in enumerate(fv_lines):
                        line = line.strip()
                        if not line: continue
                        split_cut = line.split('\t')
                        if len(split_cut) < 2:
                             print(f"Warning: Skipping malformed line {line_num+1} in {venue_file_path}")
                             continue
                        try:
                            idx1 = int(split_cut[0].strip())
                            idx2 = int(split_cut[1].strip())
                            keyi = idx_pid.get(idx1)
                            keyj = idx_pid.get(idx2)

                            if keyi is None or keyj is None:
                                 print(f"Warning: Invalid index found on line {line_num+1} in {venue_file_path}. Skipping.")
                                 continue
                                 
                            weights = 1
                            Gv.add_edge(keyi, keyj, weight=weights)
                        except ValueError:
                             print(f"Warning: Non-integer index found on line {line_num+1} in {venue_file_path}. Skipping.")
                             continue
                 except FileNotFoundError:
                      print(f"Info: Venue pair file not found for {author_name}, skipping co-venue graph edges.")
                 except Exception as e:
                      print(f"Error reading venue pair file {venue_file_path}: {e}")
            else:
                 print(f"Info: Venue pair file not found for {author_name}, skipping co-venue graph edges.")

            # Construct co-title graph
            Gt = nx.Graph()
            for pid in p_t:
                Gt.add_node(pid)
            
            for i_idx, keyi in enumerate(p_t):
                for j_idx, keyj in enumerate(p_t):
                    # Use original indices i_idx, j_idx for comparison, not pids directly
                    if j_idx > i_idx:
                         try:
                              set1 = set(p_t[keyi])
                              set2 = set(p_t[keyj])
                              weights = len(set1.intersection(set2))
                              if weights >= 2:
                                   Gt.add_edge(keyi, keyj, weight=weights)
                         except KeyError as e:
                              print(f"Warning: Key not found during title graph construction: {e}")
                              continue # Skip this pair if a key is missing
            
            # Combine graphs
            Glist = [Ga, Gt, Gv]
            
            for i in range(len(Glist)):
                for u, v, d in Glist[i].edges(data='weight'):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += d
                    else:
                        G.add_edge(u, v, weight=d)
            
            Glist.append(G)
            
            # --- Guard against empty graph for sampling --- 
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                 print(f"Warning: Combined graph G is empty or has no edges for {author_name}. Skipping sampling and training.")
                 continue
            # --- End Guard ---
                 
            # Sampling paths
            all_neighbor_samplings = []
            all_neg_sampling = []
            
            for i, Gi in enumerate(Glist):
                # Ensure Gi is not empty before converting
                if Gi.number_of_nodes() == 0:
                     # Append placeholder or handle appropriately
                     all_neighbor_samplings.append([0] * G.number_of_nodes())
                     all_neg_sampling.append([0] * G.number_of_nodes())
                     print(f"Warning: Graph G{i} is empty for {author_name}.")
                     continue 
                     
                adj_matrix = nx.to_numpy_array(Gi, nodelist=list(G.nodes())) # Ensure consistent node order
                Gtmp = copy.deepcopy(Gi)
                for u, v, d in Gtmp.edges(data='weight'):
                    Gtmp[u][v]['weight'] = 1
                
                pathl_matrix = np.full((G.number_of_nodes(), G.number_of_nodes()), np.inf) # Initialize with infinity
                np.fill_diagonal(pathl_matrix, 0) # Distance to self is 0
                
                # Calculate shortest paths only if Gtmp has nodes
                if Gtmp.number_of_nodes() > 0:
                    length = dict(nx.all_pairs_dijkstra_path_length(Gtmp))
                    node_to_idx_map = {node: idx for idx, node in enumerate(G.nodes())}

                    for u_node, paths in length.items():
                        if u_node not in node_to_idx_map: continue
                        u_idx = node_to_idx_map[u_node]
                        for v_node, dist in paths.items():
                             if v_node not in node_to_idx_map: continue
                             v_idx = node_to_idx_map[v_node]
                             pathl_matrix[u_idx, v_idx] = dist
                             # Add edges for path lengths > 0 (already handled by Dijkstra?)
                             # if not Gtmp.has_edge(u_node, v_node) and dist > 0: # This seems redundant now
                             #     Gtmp.add_edge(u_node, v_node, weight=dist)
                                 
                # pathl_matrix = nx.to_numpy_array(Gtmp, nodelist=list(G.nodes())) # Use consistent node order

                neighbor_samplings = []
                neg_samplings = []
                for node_idx_iter in range(G.number_of_nodes()): # Iterate based on G's node count
                    node_weights = adj_matrix[node_idx_iter]
                    if np.sum(node_weights) == 0:
                        neighbor_samplings.append(0)
                    else:
                        weight_distribution = node_weights / np.sum(node_weights)
                        neighbor_samplings.append(AliasSampling(weight_distribution))
                    
                    node_i_degrees = pathl_matrix[node_idx_iter].copy() # Operate on a copy
                    # node_i_degrees[node_i_degrees == 0] = 6 # Original logic - infinity is now used
                    node_i_degrees[np.isinf(node_i_degrees)] = 6 # Treat unreachable nodes with a finite large distance for sampling?
                    node_i_degrees[node_idx_iter] = 0 # Distance to self is 0
                    node_i_degrees[node_i_degrees <= 1] = 0 # Ignore direct neighbors
                    
                    if np.sum(node_i_degrees) == 0:
                        neg_samplings.append(0)
                    else:
                        node_distribution = node_i_degrees / np.sum(node_i_degrees)
                        neg_samplings.append(AliasSampling(node_distribution))
                        
                all_neighbor_samplings.append(neighbor_samplings)
                all_neg_sampling.append(neg_samplings)
            
            numwalks = 4
            walklength = 10
            negative_num = 3
            
            u_i = []
            u_j = []
            label = []
            metapath = [0, 1, 0, 2]
            
            # Perform Random Walks
            node_list = list(G.nodes()) # Get ordered list of nodes
            if not node_list: continue # Skip if graph is empty
            
            for node_idx_walk in range(G.number_of_nodes()): # Iterate by index
                for j_walk in range(0, numwalks):
                    current_node_idx = node_idx_walk
                    g_index = j_walk % len(metapath) # Cycle through metapath using modulo
                    
                    for i_walk in range(0, walklength):
                         gi = metapath[g_index]
                         # Ensure gi index is valid and samplers exist
                         if gi < len(all_neighbor_samplings) and current_node_idx < len(all_neighbor_samplings[gi]) and all_neighbor_samplings[gi][current_node_idx] != 0:
                             
                            next_node_idx = all_neighbor_samplings[gi][current_node_idx].sampling()
                            u_i.append(current_node_idx)
                            u_j.append(next_node_idx)
                            label.append(1)
                            
                            # Negative Sampling (ensure index is valid for all_neg_sampling)
                            if -1 < len(all_neg_sampling) and current_node_idx < len(all_neg_sampling[-1]) and all_neg_sampling[-1][current_node_idx] != 0:
                                for k in range(negative_num):
                                    neg_node_idx = all_neg_sampling[-1][current_node_idx].sampling()
                                    u_i.append(current_node_idx)
                                    u_j.append(neg_node_idx)
                                    label.append(-1)
                            
                            # Move to next step in metapath
                            g_index = (g_index + 1) % len(metapath)
                            gi_next = metapath[g_index]
                            
                            # Sample hop from the next node in the path
                            if gi_next < len(all_neighbor_samplings) and next_node_idx < len(all_neighbor_samplings[gi_next]) and all_neighbor_samplings[gi_next][next_node_idx] != 0:
                                
                                node_p1_idx = all_neighbor_samplings[gi_next][next_node_idx].sampling()
                                u_i.append(current_node_idx)
                                u_j.append(node_p1_idx)
                                label.append(1)
                                
                                # Negative Sampling again
                                if -1 < len(all_neg_sampling) and current_node_idx < len(all_neg_sampling[-1]) and all_neg_sampling[-1][current_node_idx] != 0:
                                     for k in range(negative_num):
                                         neg_node_idx = all_neg_sampling[-1][current_node_idx].sampling()
                                         u_i.append(current_node_idx)
                                         u_j.append(neg_node_idx)
                                         label.append(-1)
                            
                            current_node_idx = next_node_idx # Move to the next node
                         else:
                              # If no forward path, just do negative sampling
                             if -1 < len(all_neg_sampling) and current_node_idx < len(all_neg_sampling[-1]) and all_neg_sampling[-1][current_node_idx] != 0:
                                 for k in range(negative_num):
                                    neg_node_idx = all_neg_sampling[-1][current_node_idx].sampling()
                                    u_i.append(current_node_idx)
                                    u_j.append(neg_node_idx)
                                    label.append(-1)
                             # Still advance metapath index even if stuck
                             g_index = (g_index + 1) % len(metapath)
                             break # Break inner walk length loop if stuck
            
            # --- Guard against empty training data --- 
            if not u_i:
                 print(f"Warning: No training pairs generated for {author_name}. Skipping training.")
                 continue
            # --- End Guard ---
                 
            # Training
            node_attr = []
            node_map = {node: i for i, node in enumerate(G.nodes())}
            # Create attribute matrix in the same order as G.nodes()
            node_attr_np = np.zeros((G.number_of_nodes(), 100)) # Assuming 100 dim word2vec
            
            for pid_node in G.nodes():
                if pid_node in p_to:
                    words_vec = []
                    for word in p_to[pid_node]:
                        if word in model_w.wv:
                            words_vec.append(model_w.wv[word])
                    if len(words_vec) == 0:
                        # Use a consistent zero vector or random vector if title was empty/no words found
                        # words_vec.append(2 * np.random.random(100) - 1) 
                        words_vec.append(np.zeros(100))
                    # Place the attribute vector in the correct row index
                    node_idx_attr = node_map[pid_node]
                    node_attr_np[node_idx_attr] = np.mean(words_vec, 0)
                else:
                     # Handle case where pid from graph isn't in p_to (shouldn't happen ideally)
                     node_idx_attr = node_map[pid_node]
                     node_attr_np[node_idx_attr] = np.zeros(100)
                     
            node_attr = node_attr_np # Use the ordered numpy array
            
            batch_size = 64
            # Ensure total_batch is at least 1 if u_i is not empty
            total_batch = max(1, 3 * int(len(u_i) / batch_size))
            display_batch = 100
            
            # Ensure node_attr has the correct shape
            if node_attr.shape[0] != G.number_of_nodes():
                 print(f"Error: node_attr shape {node_attr.shape} mismatch with graph nodes {G.number_of_nodes()} for {author_name}")
                 continue # Skip if shapes don't match
                 
            model = GCN(Glist, node_attr, batch_size=batch_size)
            
            avg_loss = 0.
            for i in range(total_batch):
                sdx = (i * batch_size) % len(u_i)
                edx = ((i + 1) * batch_size)
                # Adjust edx for the last batch to avoid index out of bounds
                if edx > len(u_i): 
                    edx = len(u_i)
                if sdx >= edx: # Handle potential empty slice if sdx reaches end
                     continue
                
                # Use indices directly, no need for wrap-around logic if slicing correctly
                u_ii = u_i[sdx:edx]
                u_jj = u_j[sdx:edx]
                labeli = label[sdx:edx]
                
                # Ensure batch is not empty before training
                if not u_ii: 
                     continue 
                     
                loss = model.train_line(u_ii, u_jj, labeli)
                # Handle potential NaN/Inf loss
                if np.isnan(loss) or np.isinf(loss):
                     print(f"Warning: NaN or Inf loss detected at batch {i} for {author_name}. Skipping loss update.")
                     loss = 0 # Avoid propagating NaN/Inf
                     
                avg_loss += loss
                if i % display_batch == 0 and i > 0:
                    print('%d/%d loss %8.6f' % (i, total_batch, avg_loss / display_batch))
                    avg_loss = 0.
            
            # --- Guard against empty embed_matrix --- 
            try:
                embed_matrix = model.cal_embed()
                if embed_matrix is None or embed_matrix.shape[0] == 0:
                     print(f"Warning: Embedding matrix is empty for {author_name}. Skipping evaluation.")
                     continue
            except Exception as e:
                 print(f"Error calculating embeddings for {author_name}: {e}")
                 continue
            # --- End Guard ---
                 
            # Evaluating
            # Map G's internal node order (0 to N-1) back to original PIDs for GHAC
            eval_idx_pid = {idx: node for idx, node in enumerate(G.nodes())}
            num_clusters = len(set(correct_labels))
            # Ensure num_clusters is valid
            num_clusters = max(1, min(num_clusters, G.number_of_nodes()))
            
            labels = GHAC(embed_matrix, Glist[-1], eval_idx_pid, num_clusters)
            
            # Ensure labels have the correct length corresponding to correct_labels
            # We need to map the clustered nodes (which are a subset) back to the original list of publications
            # This requires careful indexing if some publications were skipped earlier.
            # For now, assuming 'labels' corresponds to nodes in embed_matrix/G.nodes(), 
            # and 'correct_labels' corresponds to the initial list of publications read.
            # This part needs careful review if publications were skipped. 
            # If we assume no publications skipped, lengths should match. 
            if len(labels) != len(correct_labels):
                 print(f"Warning: Length mismatch between predicted labels ({len(labels)}) and correct labels ({len(correct_labels)}) for {author_name}. Evaluation might be inaccurate.")
                 # Pad or truncate labels? Or skip evaluation? Skipping for now.
                 # Fallback: create dummy labels if needed for pairwise_evaluate structure
                 aligned_labels = labels[:len(correct_labels)] if len(labels) > len(correct_labels) else list(labels) + [ -1 ] * (len(correct_labels) - len(labels)) 
            else:
                 aligned_labels = labels
                 
            pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, aligned_labels)
            result.append([author_name, pairwise_precision, pairwise_recall, pairwise_f1])
            
            # --- Print Standard Mode Output (per name) --- 
            # Original competition format might have just printed these at the end? 
            # Let's print per name for clarity during run.
            print(f"Result for {author_name}: Precision={pairwise_precision:.4f}, Recall={pairwise_recall:.4f}, F1={pairwise_f1:.4f}")
            # print(f"Correct labels ({len(correct_labels)}): {correct_labels}") # Optional: Print labels if needed
            # print(f"Predicted labels ({len(aligned_labels)}): {aligned_labels}") # Optional: Print labels if needed

        # --- Print Standard Mode Average Results --- 
        if result:
            Prec = 0
            Rec = 0
            F1 = 0
            save_csvpath = 'result/'
            ensure_directory(save_csvpath) # Ensure result directory exists
            
            try:
                with open(os.path.join(save_csvpath, 'AM_nok_standard_results.csv'), 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["name", "Prec", "Rec", "F1"])
                    valid_results_count = 0
                    for i in result:
                        if isinstance(i[1], (int, float)) and isinstance(i[2], (int, float)) and isinstance(i[3], (int, float)) \
                           and not (np.isnan(i[1]) or np.isnan(i[2]) or np.isnan(i[3])):
                            Prec += i[1]
                            Rec += i[2]
                            F1 += i[3]
                            valid_results_count += 1
                        else:
                            print(f"Warning: Skipping invalid result row for {i[0]} in average calculation: {i[1:]}")
                            
                    if valid_results_count > 0:
                        Prec = Prec / valid_results_count
                        Rec = Rec / valid_results_count
                        F1 = F1 / valid_results_count
                        writer.writerow(["Avg", Prec, Rec, F1])
                        print(f"\n--- Overall Standard Mode Avg Results ({valid_results_count} names) --- ")
                        print(f"Avg Precision: {Prec:.4f}")
                        print(f"Avg Recall:    {Rec:.4f}")
                        print(f"Avg F1-Score:  {F1:.4f}")
                    else:
                         writer.writerow(["Avg", 0, 0, 0])
                         print("\n--- No valid results to average for Standard Mode --- ")
                         
                    # Write individual results
                    for i in range(len(result)):
                        tmp = result[i]
                        writer.writerow(tmp[0:4])
            except IOError as e:
                 print(f"Error writing results to CSV: {e}")
            except Exception as e:
                 print(f"Unexpected error during result writing: {e}")
        else:
             print("\n--- No results generated in Standard Mode --- ")

        print("--- Standard Mode Finished ---")
        sys.exit(0)

    # --- OpenAlex Mode (Default Behavior) --- 
    else: 
        if not args.name:
            # If name is not provided in default mode, maybe process all cached files?
            # For now, require --name for default OpenAlex mode.
            print("Error: --name argument is required when running in default OpenAlex mode.")
            print("Example: python name_disambiguation.py --name \"Author Name\"")
            print("Alternatively, run in standard mode for evaluation: python name_disambiguation.py --standard")
            sys.exit(1)
            
        print(f"--- Running in OpenAlex Mode for: {args.name} ---")
        # Process OpenAlex IDs using the specific function
        disambiguate_openAlex_ids(args.name) 
        print(f"--- Finished OpenAlex Mode for: {args.name} ---")
        sys.exit(0)

# Ensure helper functions like ensure_directory are defined if used here or imported
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)