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
    # word2vec converts
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

    result=[]

    # path is set to the directory containing the raw data files.
    path = "raw-data-temp/"
    # file_names is a list of all filenames in the specified directory.
    file_names = os.listdir(path)

    # Initialize p_t here to fix the global scope issue
    p_t = {}
    p_to = {}
    
    # Starts a loop to process each file in file_names.
    for fname in file_names:
        # Removes the last four characters from the filename, which is assumed to be the file extension (e.g., .xml).
        fname = fname[:-4]
        # Opens the file with the .xml extension in read mode with UTF-8 encoding and reads its content into the variable f.
        f = open(path + fname + ".xml",'r',encoding = 'utf-8').read()
        # Replaces all occurrences of the & character in the file content with a space to clean the text.
        text=re.sub(u"&",u" ",f)
        # Parses the cleaned XML text and gets the root element of the XML structure using the ElementTree (ET) module.
        root = ET.fromstring(text)

        correct_labels=[] # Initializes correct_labels as an empty list to store publication labels.
        p_to={} # Initializes p_to as empty dictionaries to store the original words in the title of each publication.
        p_t={}  # Initlizeis p_t as empty dictionaries to store stemmed words in the title of each publication, respectively.

        # Loops through each <publication> element in the XML root.
        for i in root.findall('publication'):
            # Extracts the text content of the <id> child element of the current <publication> element and assigns it to pid.
            pid = i.find('id').text

            # Checks if pid is already in p_t dictionary. If it is, appends '1' to pid to ensure a unique identifier.
            if pid in p_t:
                pid = pid+'1'

            # Extracts the text content of the <label> child element, converts it to an integer, and appends it to correct_labels.
            correct_labels.append(int(i.find('label').text))

            line = i.find('title').text # Extracts the text content of the <title> child element.
            line = re.sub(r, ' ', line) # Replaces all characters matching the regular expression r (punctuation and special characters) with a space.
            line = line.replace('\t',' ') # Replaces all tab characters with a space.
            line = line.lower() # Converts the title to lowercase.
            split_cut = line.split(' ') # Splits the title into a list of words using spaces as delimiters.

            p_t[pid]=[] # Initializes empty lists in p_t for the current pid.
            p_to[pid]=[] # Initializes empty lists in p_to for the current pid.

            # Iterates over each word j in the split title (split_cut)
            for j in split_cut:
                # If the length of the word is greater than 1, it is appended to the list in p_to.
                if len(j)>1:
                    p_to[pid].append(j)
                    # If the stemmed version of the word is not in the stopword list, it is appended to the list in p_t.
                    if porter_stemmer.stem(j) not in stopword:
                        p_t[pid].append(porter_stemmer.stem(j))

    # --------------------------Stemming Publication Titles-----------------------------

    # path is set to the directory containing the raw data files.
    path = "raw-data-temp/"
    # file_names is a list of all filenames in the specified directory.
    file_names = os.listdir(path)

    # Starts a loop to process each file in file_names.
    for fname in file_names:
        # Removes the last four characters from the filename, which is assumed to be the file extension (e.g., .xml).
        fname = fname[:-4]
        # Opens the file with the .xml extension in read mode with UTF-8 encoding and reads its content into the variable f.
        f = open(path + fname + ".xml",'r',encoding = 'utf-8').read()
        # Replaces all occurrences of the & character in the file content with a space to clean the text.
        text=re.sub(u"&",u" ",f)
        # Parses the cleaned XML text and gets the root element of the XML structure using the ElementTree (ET) module.
        root = ET.fromstring(text)

        correct_labels=[] # Initializes correct_labels as an empty list to store publication labels.
        p_to={} # Initializes p_to as empty dictionaries to store the original words in the title of each publication.
        p_t={}  # Initlizeis p_t as empty dictionaries to store stemmed words in the title of each publication, respectively.

        # Loops through each <publication> element in the XML root.
        for i in root.findall('publication'):
            # Extracts the text content of the <id> child element of the current <publication> element and assigns it to pid.
            pid = i.find('id').text

            # Checks if pid is already in p_t dictionary. If it is, appends '1' to pid to ensure a unique identifier.
            if pid in p_t:
                pid = pid+'1'

            # Extracts the text content of the <label> child element, converts it to an integer, and appends it to correct_labels.
            correct_labels.append(int(i.find('label').text))

            line = i.find('title').text # Extracts the text content of the <title> child element.
            line = re.sub(r, ' ', line) # Replaces all characters matching the regular expression r (punctuation and special characters) with a space.
            line = line.replace('\t',' ') # Replaces all tab characters with a space.
            line = line.lower() # Converts the title to lowercase.
            split_cut = line.split(' ') # Splits the title into a list of words using spaces as delimiters.

            p_t[pid]=[] # Initializes empty lists in p_t for the current pid.
            p_to[pid]=[] # Initializes empty lists in p_to for the current pid.

            # Iterates over each word j in the split title (split_cut)
            for j in split_cut:
                # If the length of the word is greater than 1, it is appended to the list in p_to.
                if len(j)>1:
                    p_to[pid].append(j)
                    # If the stemmed version of the word is not in the stopword list, it is appended to the list in p_t.
                    if porter_stemmer.stem(j) not in stopword:
                        p_t[pid].append(porter_stemmer.stem(j))

    # --------------------------Construct PHNet-----------------------------
    pid_idx={} # Initializes an empty dictionary pid_idx to map publication IDs (pid) to unique indices.
    idx_pid={} # Initializes an empty dictionary idx_pid to map unique indices back to publication IDs (pid).
    idx=0 # Initializes a variable idx to keep track of the unique index for each publication.

    # Creates an empty graph G using NetworkX, which will be used to construct the PHNet (Publication Heterogeneous Network).
    G = nx.Graph()

    # Starts a loop to iterate over each publication ID (pid) in the p_t dictionary. The p_t dictionary contains the processed and stemmed words for each publication ID.
    for pid in p_t:
        G.add_node(pid) # Adds the current publication ID (pid) as a node in the graph G.
        pid_idx[pid]=idx # Maps the current publication ID (pid) to the current index (idx) in the pid_idx dictionary.
        idx_pid[idx]=pid # Maps the current index (idx) back to the publication ID (pid) in the idx_pid dictionary.
        idx=idx+1 # Increments the index idx by 1 to ensure the next publication ID gets a unique index.

    # --------------------------Construct PHNet (CoAuthor Subgraph)-----------------------------

    # This line initializes an empty undirected graph Ga using the NetworkX library. This graph will be used to represent the co-author relationships.
    Ga = nx.Graph()
    # This loop iterates over each publication ID (pid) in the dictionary p_t.
    for pid in p_t:
        # For each pid, it adds a node to the graph Ga. This ensures that every publication ID is represented as a node in the graph, even if it does not end up having any edges (co-author relationships).
        Ga.add_node(pid)

    # This line opens a text file containing author pairs and reads all its lines into the list fa. Each line in the file is expected to represent a pair of authors who have co-authored a paper together.
    fa = open("experimental-results/authors/" + fname + "_authorlist.txt",'r',encoding = 'utf-8').readlines()

    # This loop iterates over each line in the list fa.
    for line in fa:
        # line.strip() removes any leading and trailing whitespace from the current line.
        line.strip()

        # line.split('\t') splits the stripped line into components based on tab characters (\t). The resulting list, split_cut, contains the different parts of the line.
        split_cut = line.split('\t')

        # This line processes the first component of split_cut, which represents the index of the publication first author.
        keyi = idx_pid[int(split_cut[0].strip())]
        # It strips any whitespace from this component, converts it to an integer, and then uses this integer to look up the corresponding publication ID (keyi) in the idx_pid dictionary.
        keyj = idx_pid[int(split_cut[1].strip())]

        # Initializes a variable weights to 1, representing the initial weight of the edge between the two publications that share an author.
        weights = 1
        if Ga.has_edge(keyi,keyj):
            # If the edge exists, increments the weight of the edge by 1.
            Ga[keyi][keyj]['weight'] = Ga[keyi][keyj]['weight'] + weights
        else:
            # If the edge does not exist, adds a new edge between keyi and keyj with an initial weight of 1.
            # Ga.add_edge(keyi,keyj,{'weight': weights})
            Ga.add_edge(keyi, keyj, weight=weights)

    # --------------------------Construct PHNet (CoVenue Subgraph)-----------------------------

    # Initializes an empty graph Gv using NetworkX. This graph will be used to represent co-venue relationships between publications.
    Gv = nx.Graph()

    # Loops through each publication ID (pid) in the p_t dictionary.
    for pid in p_t:
        # Adds each pid as a node in the graph Gv. This ensures that every publication ID is represented as a node in the graph.
        Gv.add_node(pid)

    # Opens a file that contains pairs of publication IDs that occurred in the same conference or journal, reads all lines from this file, and stores them in the list fv.
    fv = open("experimental-results/" + fname + "_jconfpair.txt",'r',encoding = 'utf-8').readlines()

    # Starts a loop to iterate over each line in the list fv.
    for line in fv:
        # Strips leading and trailing whitespace characters from the current line using line.strip(). This ensures clean data processing.
        line.strip()
        # Splits the current line into components using the tab character (\t) as the delimiter, resulting in a list split_cut.
        split_cut = line.split('\t')

        # Processes the first component of split_cut, which represents the index of the first publication ID. Strips any whitespace, converts it to an integer, and uses this integer to look up the corresponding publication ID (keyi) in the idx_pid dictionary.
        keyi = idx_pid[int(split_cut[0].strip())]
        # Processes the second component of split_cut, which represents the index of the second publication ID. Strips any whitespace, converts it to an integer, and uses this integer to look up the corresponding publication ID (keyj) in the idx_pid dictionary.
        keyj = idx_pid[int(split_cut[1].strip())]
        weights = 1
        # Gv.add_edge(keyi,keyj,{'weight': weights})
        # Adds an edge between keyi and keyj with an initial weight of 1 in the graph Gv.
        Gv.add_edge(keyi, keyj, weight=weights)

    # --------------------------Construct PHNet (CoTitle Subgraph)-----------------------------

    # Initializes an empty graph Gt using NetworkX. This graph will be used to represent the co-title relationships between publications.
    Gt = nx.Graph()
    # Loops through each publication ID (pid) in the p_t dictionary.
    for pid in p_t:
        # Adds each pid as a node in the graph Gt. This ensures that every publication ID is represented as a node in the graph.
        Gt.add_node(pid)

    # These nested loops iterate over each pair of publication IDs (keyi and keyj) in the p_t dictionary.
    for i, keyi in enumerate(p_t):
        for j, keyj in enumerate(p_t):
            # Converts the list of stemmed words for each publication (p_t[keyi] and p_t[keyj]) into sets.
            # Computes the intersection of these two sets, which gives the common stemmed words between the two publications.
            # The length of this intersection is assigned to the variable weights, representing the number of shared stemmed words in the titles of the two publications.
            weights=len(set(p_t[keyi]).intersection(set(p_t[keyj])))
            # Checks if the current index j is greater than the current index i to avoid duplicate edges and self-loops.
            # Also checks if the weights (number of shared words) is greater than or equal to 2. This ensures that edges are only added for pairs of publications with at least two common words in their titles.
            if (j>i and weights>=2):
                # Gt.add_edge(keyi,keyj,{'weight': weights})
                # Adds an edge between keyi and keyj in the graph Gt with the computed weight.
                Gt.add_edge(keyi, keyj, weight=weights)


    Glist=[]
    Glist.append(Ga)
    Glist.append(Gt)
    Glist.append(Gv)

    # Combine edges and weights from all graphs into the final graph G
    for i in range(len(Glist)):
        # For each graph Glist[i], iterates over its edges. The edges(data='weight') method returns tuples of the form (u, v, d), where u and v are the nodes connected by the edge, and d is the weight of the edge.
        for u, v, d in Glist[i].edges(data='weight'):
            if G.has_edge(u, v):
                G[u][v]['weight'] += d  # If the edge already exists, it increments the weight of the edge in G by d, the weight from the current graph.
            else:
                G.add_edge(u, v, weight=d)  # If the edge does not exist in G, it adds a new edge between u and v with the weight d.
    # Append the final combined graph G to the list Glist.
    Glist.append(G)

    # --------------------------Sampling Paths via Meta-path and Relation Weight Guided Random Walks:-----------------------------

    all_neighbor_samplings=[] # Store sampling distributions for neighbor nodes
    all_neg_sampling=[] # Store negative samples


    # Starts a loop that iterates over each graph Gi in the list Glist. The variable i is the index of the graph.
    for i,Gi in enumerate(Glist):
        # Converts the graph Gi to a NumPy array representing its adjacency matrix. This matrix contains the edge weights between nodes.
        adj_matrix = nx.to_numpy_array(Gi)
        # Creates a deep copy of the graph Gi and stores it in Gtmp.
        Gtmp= copy.deepcopy(Gi)
        # Loops through each edge (u, v, d) in Gtmp, where u and v are nodes and d is the edge weight. Sets the weight of each edge to 1.
        for u,v,d in Gtmp.edges(data = 'weight'):
            Gtmp[u][v]['weight'] = 1
        # Uses Dijkstra's algorithm to calculate the shortest path lengths between all pairs of nodes in Gtmp. The result is stored in the dictionary length.
        length = dict(nx.all_pairs_dijkstra_path_length(Gtmp))

        # Loops through each pair of nodes (u, v) in the length dictionary.
        for u in length:
            for v in length[u]:
                # If there is no direct edge between u and v in Gtmp and the shortest path length between them is greater than 0, adds an edge with the shortest path length as the weight.
                if Gtmp.has_edge(u,v) is False and length[u][v]>0:
                    Gtmp.add_edge(u, v, weight=length[u][v])
        # Converts the updated graph Gtmp to a NumPy array representing its adjacency matrix, where the weights represent shortest path lengths.
        pathl_matrix = nx.to_numpy_array(Gtmp)

        # Store sampling distributions for the current graph.
        neighbor_samplings = []
        neg_samplings=[]
        # Loops through each node i in the graph
        for i in range(G.number_of_nodes()):
            # Gets the edge weights for node i from adj_matrix.
            node_weights = adj_matrix[i]
            # If the sum of weights is 0, appends 0 to neighbor_samplings.
            if np.sum(node_weights)==0:
                neighbor_samplings.append(0)
            # Otherwise, normalizes the weights to create a probability distribution and appends an AliasSampling object initialized with this distribution to neighbor_samplings.
            else :
                weight_distribution = node_weights / np.sum(node_weights)
                neighbor_samplings.append(AliasSampling(weight_distribution))

            # Gets the shortest path lengths for node i from pathl_matrix.
            node_i_degrees = pathl_matrix[i]
            # Sets zero weights to 6, the weight for the node itself to 0, and any weights less than or equal to 1 to 0.
            node_i_degrees[node_i_degrees==0] = 6
            node_i_degrees[i]=0
            node_i_degrees[node_i_degrees<=1] = 0

            # If the sum of the adjusted weights is 0, appends 0 to neg_samplings.
            if np.sum(node_i_degrees)==0:
                neg_samplings.append(0)
            # Otherwise, normalizes the weights to create a probability distribution and appends an AliasSampling object initialized with this distribution to neg_samplings.
            else:
                node_distribution = node_i_degrees / np.sum(node_i_degrees)
                neg_samplings.append(AliasSampling(node_distribution))
        # Appends the neighbor_samplings and neg_samplings lists to all_neighbor_samplings and all_neg_sampling, respectively.
        all_neighbor_samplings.append(neighbor_samplings)
        all_neg_sampling.append(neg_samplings)

    numwalks=4 # Number of random walks to start from each node.
    walklength=10 # Length of each random walk.
    negative_num=3 # Number of negative samples to generate per positive sample.

    # Initializes empty lists u_i, u_j, and label to store the node pairs and labels for training data.
    u_i=[]
    u_j=[]
    label=[]
    # Defines a metapath for the walks, specifying the sequence of graphs to use in the walks.
    metapath=[0,1,0,2]

    # Perform Random Walks

    # Loops through each node node_index in the graph:
    for node_index in range(G.number_of_nodes()):
        # Starts numwalks random walks from the current node.
        # For each walk, initializes the starting node and sets the graph index based on the metapath.
        for j in range(0, numwalks):
            node_start=node_index
            g_index=j
            gi=metapath[g_index]
            # For each step in the walk
            for i in range(0, walklength):
                if all_neighbor_samplings[gi][node_start] != 0:
                    # Samples a neighbor node from the current graph using the sampling distribution.
                    node_p = all_neighbor_samplings[gi][node_start].sampling()
                    u_i.append(node_start)
                    u_j.append(node_p)
                    # Records the pair (node_start, node_p) with a positive label (1).
                    label.append(1)

                    # Negative Sampling

                    # Samples negative_num negative nodes and records the pairs (node_start, node_n) with negative labels (-1).
                    if all_neg_sampling[-1][node_start] != 0:
                        for k in range(negative_num):
                            node_n = all_neg_sampling[-1][node_start].sampling()
                            u_i.append(node_start)
                            u_j.append(node_n)
                            label.append(-1)

                    # Updates the graph index and graph for the next step.
                    g_index=(g_index+1)%len(metapath)
                    gi=metapath[g_index]

                    # Samples the next node and records the pair with a positive label (1).
                    if all_neighbor_samplings[gi][node_p] != 0:

                        node_p1 = all_neighbor_samplings[gi][node_p].sampling()
                        u_i.append(node_start)
                        u_j.append(node_p1)
                        label.append(1)

                        # Performs negative sampling as before.
                        if all_neg_sampling[-1][node_start] != 0:
                            for k in range(negative_num):
                                node_n = all_neg_sampling[-1][node_start].sampling()
                                u_i.append(node_start)
                                u_j.append(node_n)
                                label.append(-1)

                    # Updates the starting node for the next step.
                    node_start = node_p
                # If no neighbors are available, performs negative sampling directly.
                else:
                    for k in range(negative_num):
                        node_n = all_neg_sampling[-1][node_start].sampling()
                        u_i.append(node_start)
                        u_j.append(node_n)
                        label.append(-1)
                    # Updates the graph index and graph for the next step.
                    g_index=(g_index+1)%len(metapath)
                    gi=metapath[g_index]

    # Training
    node_attr = []
    for pid in p_to:
        words_vec=[]
        for word in p_to[pid]:
            # if (word in model_w):
            if word in model_w.wv:
                # words_vec.append(model_w[word])
                 words_vec.append(model_w.wv[word])
        if len(words_vec)==0:
            words_vec.append(2*np.random.random(100)-1)
        node_attr.append(np.mean(words_vec,0))
    node_attr=np.array(node_attr)

    batch_size = 64
    total_batch = 3*int(len(u_i)/batch_size)
    display_batch = 100

    model = GCN(Glist, node_attr, batch_size=batch_size)

    avg_loss = 0.
    for i in range(total_batch):
        sdx=(i*batch_size)%len(u_i)
        edx=((i+1)*batch_size)%len(u_i)
        #print (sdx,edx)
        if edx>sdx:
            u_ii = u_i[sdx:edx]
            u_jj = u_j[sdx:edx]
            labeli = label[sdx:edx]
        else:
            u_ii = u_i[sdx:]+u_i[0:edx]
            u_jj = u_j[sdx:]+u_j[0:edx]
            labeli = label[sdx:]+label[0:edx]
        loss= model.train_line(u_ii, u_jj, labeli)
        avg_loss += loss / display_batch
        if i % display_batch == 0 and i > 0:
            print ('%d/%d loss %8.6f' %(i,total_batch,avg_loss))
            avg_loss = 0.



    # Evaluating
    embed_matrix = model.cal_embed()
    labels = GHAC(embed_matrix,Glist[-1],idx_pid,len(set(correct_labels)))
    pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels,labels)
    result.append([fname,pairwise_precision, pairwise_recall, pairwise_f1])
    
    # Comment out or replace with this to hide original output:
    # print_original_output(fname, correct_labels, labels, pairwise_precision, pairwise_recall, pairwise_f1)
    
    # Only show the custom output
    generate_author_id_clusters(fname, correct_labels, labels)

    # Macro-F1
    Prec = 0
    Rec = 0
    F1 = 0
    save_csvpath = 'result/'
    with open(save_csvpath+'AM_nok.csv','w',newline='',encoding = 'utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name","Prec","Rec","F1"])
        for i in result:
            Prec = Prec + i[1]
            Rec = Rec + i[2]
            F1 = F1 + i[3]
        Prec = Prec/len(result)
        Rec = Rec/len(result)
        F1 = F1/len(result)
        writer.writerow(["Avg",Prec,Rec,F1])
        for i in range(len(result)):
            tmp = result[i]
            writer.writerow(tmp[0:4])

    print ("Avg",Prec,Rec,F1)