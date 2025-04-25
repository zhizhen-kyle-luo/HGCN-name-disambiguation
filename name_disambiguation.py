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
    
    # Determine optimal number of clusters based on the number of author IDs
    # Use at least 2 clusters, at most the number of author IDs
    n_clusters = max(2, min(len(author_ids), 10))
    
    # Perform clustering
    labels = GHAC(embed_matrix, Glist[-1], idx_pid, n_clusters)
    
    # Create mapping from publication index to cluster
    pub_clusters = {}
    for i, cluster_id in enumerate(labels):
        if i < len(publication_to_author_id):
            author_id = publication_to_author_id.get(i)
            if author_id:
                if cluster_id not in pub_clusters:
                    pub_clusters[cluster_id] = []
                pub_clusters[cluster_id].append(author_id)
    
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
    
    # Print result
    print(f"\n{fname} OpenAlex author ID clusters:")
    print(json.dumps({fname: person_clusters}, indent=2))
    
    return person_clusters

if __name__ == '__main__':
    # Check if we're in OpenAlex disambiguation mode
    if len(sys.argv) > 1 and sys.argv[1] == "--openAlex":
        # Get the author name from command line or use a default
        author_name = sys.argv[2] if len(sys.argv) > 2 else "Russell Bowler"
        # Process OpenAlex IDs
        disambiguate_openAlex_ids(author_name)
        sys.exit(0)

    # Pre-trained word2vec model
    save_model_name = "gene/word2vec.model"
    model_w = word2vec.Word2Vec.load(save_model_name)

    # This line defines a regular expression pattern r that matches one or more occurrences of a variety of punctuation and special characters.
    r = '[!""#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'
    # This line defines a list of common stopwords. Stopwords are words that are often filtered out during text processing because they are considered to have little value in text analysis.
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','method','algrithom','by','model']
    # This line stems each word in the stopword list using the Porter stemmer. Stemming is the process of reducing a word to its root form. For example, 'using' might be reduced to 'use'.
    # porter_stemmer.stem(w) applies the stemming algorithm to each word w in the stopword list.
    # The list comprehension [porter_stemmer.stem(w) for w in stopword] creates a new list where each stopword has been stemmed.
    stopword = [porter_stemmer.stem(w) for w in stopword]

    result = []
    path = "raw-data-temp/"
    file_names = os.listdir(path)
    
    # Process each file individually
    for fname in file_names:
        if not fname.endswith('.xml'):
            continue
            
        # Extract name without extension
        author_name = fname[:-4]
        print(f"\nProcessing {author_name}...")
        
        # Initialize variables for this author
        p_t = {}
        p_to = {}
        correct_labels = []
        
        # Read and process XML file
        f = open(path + fname, 'r', encoding = 'utf-8').read()
        text = re.sub(u"&", u" ", f)
        root = ET.fromstring(text)

        # Process publications
        for i in root.findall('publication'):
            pid = i.find('id').text
            
            if pid in p_t:
                pid = pid + '1'
                
            correct_labels.append(int(i.find('label').text))
            
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
        
        # Construct co-author graph
        Ga = nx.Graph()
        for pid in p_t:
            Ga.add_node(pid)
        
        author_file_path = f"experimental-results/authors/{author_name}_authorlist.txt"
        if os.path.exists(author_file_path):
            fa = open(author_file_path, 'r', encoding='utf-8').readlines()
            
            for line in fa:
                line = line.strip()
                split_cut = line.split('\t')
                
                keyi = idx_pid[int(split_cut[0].strip())]
                keyj = idx_pid[int(split_cut[1].strip())]
                
                weights = 1
                if Ga.has_edge(keyi, keyj):
                    Ga[keyi][keyj]['weight'] = Ga[keyi][keyj]['weight'] + weights
                else:
                    Ga.add_edge(keyi, keyj, weight=weights)
        
        # Construct co-venue graph
        Gv = nx.Graph()
        for pid in p_t:
            Gv.add_node(pid)
        
        venue_file_path = f"experimental-results/{author_name}_jconfpair.txt"
        if os.path.exists(venue_file_path):
            fv = open(venue_file_path, 'r', encoding='utf-8').readlines()
            
            for line in fv:
                line = line.strip()
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
        
        numwalks = 4
        walklength = 10
        negative_num = 3
        
        u_i = []
        u_j = []
        label = []
        metapath = [0, 1, 0, 2]
        
        # Perform Random Walks
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
        
        # Training
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
        
        # Evaluating
        embed_matrix = model.cal_embed()
        labels = GHAC(embed_matrix, Glist[-1], idx_pid, len(set(correct_labels)))
        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, labels)
        result.append([author_name, pairwise_precision, pairwise_recall, pairwise_f1])
        
        # Generate and output the clusters for this author
        generate_author_id_clusters(author_name, correct_labels, labels)
    
    # Output overall results
    if result:
        Prec = 0
        Rec = 0
        F1 = 0
        save_csvpath = 'result/'
        with open(save_csvpath + 'AM_nok.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "Prec", "Rec", "F1"])
            for i in result:
                Prec = Prec + i[1]
                Rec = Rec + i[2]
                F1 = F1 + i[3]
            Prec = Prec / len(result)
            Rec = Rec / len(result)
            F1 = F1 / len(result)
            writer.writerow(["Avg", Prec, Rec, F1])
            for i in range(len(result)):
                tmp = result[i]
                writer.writerow(tmp[0:4])
        
        print("Avg", Prec, Rec, F1)