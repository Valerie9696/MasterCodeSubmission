import graphviz as gv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import panel as pn
import pickle as pkl
import torch

import config as cfg

evidences_with_names = pd.read_csv(os.path.join(cfg.DATA_DIR, 'variables_en.csv'))
values = pd.read_csv(os.path.join(cfg.DATA_DIR, 'values.csv'), index_col='Value')

def make_pie_chart(pathology_name, values, path):
        labels = [v for v in values.keys() if values[v][0] > 0]
        sizes = [v[0] for v in values.values() if v[0] > 0]
        colors = [v[1] / v[0] for v in values.values() if v[0] > 0]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.Blues
        normalized_colors = [cmap(norm(color)) for color in colors]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'aspect': 'equal'})
        wedge_props = {'edgecolor': 'black', 'linewidth': 1}
        ax.pie(sizes, labels=labels, colors=normalized_colors, autopct='%1.1f%%',startangle=100, wedgeprops=wedge_props, labeldistance=1.75, pctdistance=1.3, textprops={'fontsize': 14}, radius=0.6)
        ax.set_aspect('equal')
        #plt.title(pathology_name)
        plt.savefig(path, format='png', dpi=300)#, bbox_inches='tight')
        plt.close('all')
        """
        only used once to plot and show blue color scale and screenshot it 
        fig, ax = plt.subplots(figsize=(1, 3))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        color_bar = plt.colorbar(sm, cax=ax, orientation='vertical')
        color_bar.set_label('Color Scale:\n The darker the blue of shade, the higher the importance for the diagnosis.', fontsize=12)
        plt.close('all')
        """

def make_bar_chart(pathology_name, values, path):
    plt.rcParams['font.size'] = 16
    labels, col_values, colors = [], [], []
    # no iteration over value.keys() in order to keep the right order of age groups
    for key in ['age <1', ' age 1-4', 'age 5-14', ' age 15-29', ' age 30-44', ' age 45-59', ' age 60-74',' age >75']:
        v = values[key]
        if v[0] > 0:
            labels.append(key.replace('age', '').replace(' ', ''))
            colors.append(v[1]/v[0])
            col_values.append(v[0])
    total = sum(col_values)
    col_percent = [(size / total) * 100 for size in col_values]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.Blues
    normalized_colors = [cmap(norm(color)) for color in colors]
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, col_percent, color=normalized_colors, edgecolor='black', width=0.4)
    if len(col_percent) > 0:
        ax.set_ylim(0, max(col_percent) * 1.1)
    for i, (bar, size) in enumerate(zip(bars, col_percent)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{size:.1f}%', ha='center', va='bottom', color='black', fontsize=12)
    ax.set_ylabel('Number of people in %', labelpad=10)
    ax.set_xlabel('Age Groups', labelpad=10)
    ax.set_ylim(0, 100)
    plt.subplots_adjust(left=0.16, bottom=0.13)
    plt.savefig(path, format='png', dpi=300)
    #plt.show()
    plt.close('all')


def read_file(path, mode, placeholder):
    try:
        with open(path, mode) as file:
            content = file.read()
            file.close()
    except:
        print(path)
        print('File does not exist.')
        content = placeholder
    return content


def insert_line_breaks(sentence, max_length=15):
    """
    Insert linebreaks to format the questions belonging to the evidences.
    :param sentence: question belonging to the evidence
    :param max_length: line length within a node
    :return: the question with linebreaks inserted
    """
    words = sentence.split()
    current_line = ""
    result = []
    for word in words:
        if len(current_line) + len(word) + 1 > max_length:  # add words until max_length is reached
            result.append(current_line.strip())
            current_line = word + " "
        else:
            current_line += word + " "

    if current_line:
        result.append(current_line.strip())

    joined_lines = "\n".join(result)  # join the lines with a linebreak between them

    return joined_lines


def get_brightness(rgb_value):
    """
    relative luminance; formula taken from https://www.w3.org/WAI/GL/wiki/Relative_luminance
    :param rgb_value: rgb value, whose brightness is supposed to be calculated
    :return:
    """
    r, g, b, _ = rgb_value
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return brightness


def plot_ddx_attention_graph(en_in, de_in, input_vocab, output_vocab, en_attention_weights, de_attention_weights, pathology, counter):
    """
    Plot the graphs of the differential diagnoses with nodes colored by attention weights of encoder and decoder
    and save them.
    :param en_in: input of the encoder
    :param de_in: input of the decoder
    :param input_vocab: vocabulary of encoder
    :param output_vocab: vocabulary of decoder
    :param en_attention_weights: attention weights of encoder
    :param de_attention_weights: attention weights of decoder
    :param pathology: the predicted pathology
    :param counter: number of patient in the test set (only needed for the file name)
    :return:
    """
    print(counter, ' ddx_graph')
    evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
    graph = nx.DiGraph()
    color_map = {}
    ddx_nodes = []
    ddx_aws = []
    for idx, token in enumerate(de_in):
        ddx = output_vocab[token.item()]
        if ddx not in ['<bos>', '<eos>', '<pad>']:
            ddx_nodes.append(ddx)
            ddx_aws.append(de_attention_weights[idx])
            color_map[ddx] = de_attention_weights[idx].mean().item() # marker
    with open(os.path.join(cfg.DATA_DIR, cfg.VALUES), 'rb') as pickle_file:
        value_meanings = pkl.load(pickle_file)
    antecedents = []
    symptoms = []
    nodes = []
    title = "Patient: "
    if len(ddx_aws) > 1:
        ddx_aws = torch.stack(ddx_aws)
        de_aws = torch.mean(ddx_aws, dim=0)
    elif len(ddx_aws) == 1:
        de_aws = ddx_aws[0]
    else:
        return None
    for idx, ei in enumerate(en_in):
        ei = ei.item()
        aw = de_aws[idx]
        if ei != 0:
            ev = input_vocab[ei]
            if ev not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', '<edge>'] and 'edgetoken' not in ev:
                if ev not in ['F', 'M', '_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59',
                           '_age_60-74', '_age_>75']:
                    if 'V_' in ev or ev in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'False', 'True']:
                        if 'V_' in ev:
                            value_dict = value_meanings[ev]
                            value = value_dict['en']
                            if value.endswith("(R)"):
                                value = "right " + value[:-3].strip()
                            elif value.endswith("(L)"):
                                value = "left " + value[:-3].strip()
                            if value == 'N':
                                value = 'No.'
                        else:
                            value = ev
                        if value == 'No.':
                            prev_ev = input_vocab[en_in[idx - 1].item()]
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            ev_question = evidences[prev_ev]['question_en']
                            ev_question = insert_line_breaks(ev_question)
                            ev_q = ev_question.replace(":", ';')
                            ev_q = insert_line_breaks(ev_q)
                            ev_q = ev_name + '\n\n' + ev_q + '\n \n No.'
                            nodes = [ev_q if ev_question in item else item for item in nodes]
                            antecedents = [ev_q if ev_question in item else item for item in antecedents]
                            symptoms = [ev_q if ev_question in item else item for item in symptoms]
                            rename_mapping = {ev_question: ev_q}
                            graph = nx.relabel_nodes(graph, rename_mapping)
                            color_map[ev_q] = aw.item()
                        else:
                            graph.add_node(value)
                            nodes.append(value)
                            prev_ev = input_vocab[en_in[idx-1].item()]
                            ev_question = evidences[prev_ev]['question_en']
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            connect_to = ev_question.replace(":", ';')
                            connect_to = insert_line_breaks(connect_to)
                            connect_to = ev_name + '\n\n' + connect_to
                            color_map[ev_question] = aw.item()
                            graph.add_edge(connect_to, value)
                            color_map[value] = aw.item()
                    else:
                        is_antecedent = evidences[ev]['is_antecedent']
                        ev_question = evidences[ev]['question_en']
                        ev_name = evidences_with_names.loc[evidences_with_names['Code'] == ev, 'Variable'].values[0]
                        ev_name = ev_name.replace(":", " = ")
                        ev_name = insert_line_breaks(ev_name)
                        ev_question = ev_question.replace(":", ';')
                        ev_question = insert_line_breaks(ev_question)
                        ev_question = ev_name + '\n\n' + ev_question
                        ev_type = evidences[ev]['data_type']
                        if ev_type == 'B':
                            ev_question += '\n \n Yes.'
                        color_map[ev_question] = aw.item()
                        if is_antecedent:
                            antecedents.append(ev_question)
                        else:
                            symptoms.append(ev_question)
                        nodes.append(ev_question)
                else:
                    if ev == 'M':
                        title += 'Male, '
                    elif ev == 'F':
                        title += 'Female, '
                    elif ev in ['_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59', '_age_60-74', '_age_>75']:
                        text = ev.replace('_', ' ')
                        title += text
    nx_graph = nx.drawing.nx_pydot.to_pydot(graph)
    dot = gv.Digraph()
    dot.attr(compound='true')           # necessary for the subgraph

    for node in nx_graph.get_nodes():
        node_name = node.get_name().strip('"')
        if node_name not in ('graph', 'node', 'edge'):
            dot.node(node_name, fontsize='42')

    for edge in nx_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        dot.edge(src, dst, penwidth='2')

    subgraph_name = 'cluster_ddxs'
    anchor_node = 'container_anchor'
    anchor_set = False
    with dot.subgraph(name=subgraph_name) as ddxs:
        ddxs.attr(color='blue', margin='16', penwidth='2')

        center_pos = 0
        if len(ddx_nodes) < 10 and len(ddx_nodes) > 1:
            center_pos = int(len(ddx_nodes)/2)-1
        elif len(ddx_nodes) == 1:
            center_pos = 1#

        for idx, node in enumerate(ddx_nodes):
            ddxs.node(node)
            nodes.append(node)
            if idx == 10:
                break
            if idx == center_pos:
                ddxs.node(anchor_node, shape='point', style='invis', layer="-1", margin="2.5")      # insert the invisible, point shaped anchor node
                anchor_set = True

        if not anchor_set:
            ddxs.node(anchor_node, shape='point', style='invis', layer="-1", margin="2.5")

    ddxs_edge_drawn = [] # there are duplicates in the list, keep track in order to avoid drawing multiple edges between the same nodes
    for node in antecedents:
        if node not in ddxs_edge_drawn:
            dot.edge(node, anchor_node, lhead=subgraph_name, penwidth='2')
            ddxs_edge_drawn.append(node)

    for node in symptoms:
        if node not in ddxs_edge_drawn:
            dot.edge(anchor_node, node, ltail=subgraph_name, penwidth='2')
            ddxs_edge_drawn.append(node)

    dot.attr(ranksep='4', nodesep='0.5')  # higher distance between nodes, which keeps edges from overlapping (mostly)
    dot.attr(splines='true')  # use splines, so edges won't go through nodes
    dot.attr(overlap='false')

    norm = mcolors.Normalize(vmin=0, vmax=1)

    cmap = plt.cm.Blues
    for node in nodes:
        #print(node)
        node_name = node
        if node_name in color_map:
            color_value = cmap(norm(color_map[node_name]))
            color_hex = mcolors.to_hex(color_value)
            brightness = get_brightness(color_value)
            if brightness < 0.4:
                fontcolor = 'white'
            else:
                fontcolor = 'black'
            dot.node(node_name, style='filled', fillcolor=color_hex, fontcolor=fontcolor, fontsize = '42', penwidth='2')
        else:
            print(node_name)
        if node_name == pathology:
            dot.node(node_name, penwidth='5',fontsize='42')

    gv_graph = dot

    description_file = os.path.join(cfg.EX_PATH, f"{counter}.txt")
    with open(description_file, 'r') as f:
        description_text = f.read()
        f.close()

    gv_graph.attr(xlabel=description_text, labelloc='b', fontsize='12', align='left')

    if cfg.DE_ATTENTION_WEIGHTS_TYPE == 0:
        save_path = os.path.join(cfg.PLOTS_PATH, 'sum', 'ddx_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 1:
        save_path = os.path.join(cfg.PLOTS_PATH, 'mean', 'ddx_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 2:
        save_path = os.path.join(cfg.PLOTS_PATH, 'max', 'ddx_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 3:
        save_path = os.path.join(cfg.PLOTS_PATH, 'full', 'ddx_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    else:
        print('Please set ATTENTION_WEIGHTS_TYPE variable in config.py to either 0, 1 or 2.')


def plot_pathology_attention_graph(en_in, de_in, input_vocab, output_vocab, path, de_attention_weights, pathology, counter, save_name):
    """
    Plot the graphs of the differential diagnoses with nodes colored by attention weights of encoder and decoder
    and save them.
    :param en_in: input of the encoder
    :param de_in: input of the decoder
    :param input_vocab: vocabulary of encoder
    :param output_vocab: vocabulary of decoder
    :param en_attention_weights: attention weights of encoder
    :param de_attention_weights: attention weights of decoder
    :param pathology: the predicted pathology
    :param counter: number of patient in the test set (only needed for the file name)
    :return:
    """
    evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
    graph = nx.DiGraph()
    color_map = {}
    for idx, token in enumerate(de_in):
        ddx = output_vocab[token.item()]
        if ddx == pathology:
            pathology_idx = idx

    with open(os.path.join(cfg.DATA_DIR, cfg.VALUES), 'rb') as pickle_file:
        value_meanings = pkl.load(pickle_file)
    antecedents = []
    symptoms = []
    nodes = []
    de_aws = de_attention_weights[pathology_idx]
    for idx, ei in enumerate(en_in):
        ei = ei.item()
        aw = de_aws[idx]
        if ei != 0:
            ev = input_vocab[ei]
            if ev not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', '<edge>'] and 'edgetoken' not in ev:
                if ev not in ['F', 'M', '_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59',
                           '_age_60-74', '_age_>75']:
                    if 'V_' in ev or ev in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'False', 'True']:
                        if 'V_' in ev:
                            value_dict = value_meanings[ev]
                            value = value_dict['en']
                            if value.endswith("(R)"):
                                value = "right " + value[:-3].strip()
                            elif value.endswith("(L)"):
                                value = "left " + value[:-3].strip()
                            if value == 'N':
                                value = 'No.'
                        else:
                            value = ev
                        if value == 'No.':
                            prev_ev = input_vocab[en_in[idx - 1].item()]
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            ev_question =  ev_name
                            ev_q = ev_question.replace(":", '\n')
                            ev_q = insert_line_breaks(ev_q)
                            ev_q = ev_q + '\n \n No.'
                            nodes = [ev_q if ev_question in item else item for item in nodes]
                            antecedents = [ev_q if ev_question in item else item for item in antecedents]
                            symptoms = [ev_q if ev_question in item else item for item in symptoms]
                            rename_mapping = {ev_question: ev_q}
                            graph = nx.relabel_nodes(graph, rename_mapping)
                            color_map[ev_q] = aw.item()
                        else:
                            graph.add_node(value)
                            nodes.append(value)
                            prev_ev = input_vocab[en_in[idx-1].item()]
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            connect_to = ev_name
                            color_map[ev_name] = aw.item()
                            graph.add_edge(connect_to, value)
                            color_map[value] = aw.item()
                    else:
                        is_antecedent = evidences[ev]['is_antecedent']
                        ev_name = evidences_with_names.loc[evidences_with_names['Code'] == ev, 'Variable'].values[0]
                        ev_name = ev_name.replace(":", " = ")
                        ev_name = insert_line_breaks(ev_name)
                        ev_question = ev_name
                        ev_type = evidences[ev]['data_type']
                        if ev_type == 'B':
                            ev_question += '\n \n Yes.'
                        color_map[ev_question] = aw.item()
                        if is_antecedent:
                            antecedents.append(ev_question)
                            nodes.append(ev_question)
                        else:
                            symptoms.append(ev_question)
                            nodes.append(ev_question)

    nx_graph = nx.drawing.nx_pydot.to_pydot(graph)
    dot = gv.Digraph()
    for node in nx_graph.get_nodes():
        node_name = node.get_name().strip('"')
        if node_name not in ('graph', 'node', 'edge'):
            dot.node(node_name, fontsize='42')

    for edge in nx_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        dot.edge(src, dst, penwidth='2')

    ddxs_edge_drawn = []        # there are duplicates in the list, keep track in order to avoid drawing multiple edges between the same nodes
    for node in antecedents:
        if node not in ddxs_edge_drawn:
            dot.edge(node, pathology, penwidth='2')
            ddxs_edge_drawn.append(node)

    for node in symptoms:
        if node not in ddxs_edge_drawn:
            dot.edge(pathology, node, penwidth='2')
            ddxs_edge_drawn.append(node)

    dot.attr(ranksep='4', nodesep='0.5')  # higher distance between nodes, which keeps edges from overlapping (mostly)
    dot.attr(splines='true')  # use splines, so edges won't go through nodes
    norm = mcolors.Normalize(vmin=0, vmax=1)

    cmap = plt.cm.Blues
    for node in nodes:
        node_name = node
        if node_name in color_map:
            color_value = cmap(norm(color_map[node_name]))
            color_hex = mcolors.to_hex(color_value)
            brightness = get_brightness(color_value)
            if brightness < 0.4:
                fontcolor = 'white'
            else:
                fontcolor = 'black'
            dot.node(node_name, style='filled', fillcolor=color_hex, fontcolor=fontcolor, penwidth='2', fontsize='42')
        if node_name == pathology:
            dot.node(node_name, penwidth='5', fontsize='42')
    dot.node(pathology, penwidth='2', fontsize='42')
    gv_graph = dot

    description_file = os.path.join(cfg.EX_PATH, f"{counter}.txt")
    with open(description_file, 'r') as f:
        description_text = f.read()
        f.close()
    gv_graph.attr(xlabel=description_text, labelloc='b', fontsize='12', align='left')
    gv_graph.render(filename=save_name, directory=path, format='png', cleanup=True)


def plot_path_ddx_attention_graphs(en_in, de_in, input_vocab, output_vocab, de_attention_weights, pathology, counter):
    """
    Plot the graphs of the differential diagnoses with nodes colored by attention weights of encoder and decoder
    and save them.
    :param en_in: input of the encoder
    :param de_in: input of the decoder
    :param input_vocab: vocabulary of encoder
    :param output_vocab: vocabulary of decoder
    :param en_attention_weights: attention weights of encoder
    :param de_attention_weights: attention weights of decoder
    :param pathology: the predicted pathology
    :param counter: number of patient in the test set (only needed for the file name)
    :return:
    """

    evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
    graph = nx.DiGraph()
    color_map = {}
    ddx_nodes = []
    for idx, token in enumerate(de_in):
        ddx = output_vocab[token.item()]
        if ddx == pathology:
            pathology_idx = idx
        if ddx not in ['<bos>', '<eos>', '<pad>']:
            ddx_nodes.append(ddx)
            a= de_attention_weights[idx]
            color_map[ddx] = de_attention_weights[idx].mean().item() # marker
    with open(os.path.join(cfg.DATA_DIR, cfg.VALUES), 'rb') as pickle_file:
        value_meanings = pkl.load(pickle_file)
    antecedents = []
    symptoms = []
    nodes = []
    de_aws = de_attention_weights[pathology_idx]
    for idx, ei in enumerate(en_in):
        ei = ei.item()
        aw = de_aws[idx]
        if ei != 0:
            ev = input_vocab[ei]
            if ev not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', '<edge>'] and 'edgetoken' not in ev:
                if ev not in ['F', 'M', '_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59',
                           '_age_60-74', '_age_>75']:
                    if 'V_' in ev or ev in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'False', 'True']:
                        if 'V_' in ev:
                            value_dict = value_meanings[ev]
                            value = value_dict['en']
                            if value.endswith("(R)"):
                                value = "right " + value[:-3].strip()
                            elif value.endswith("(L)"):
                                value = "left " + value[:-3].strip()
                            if value == 'N':
                                value = 'No.'
                        else:
                            value = ev
                        if value == 'No.':
                            prev_ev = input_vocab[en_in[idx - 1].item()]
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            ev_question = evidences[prev_ev]['question_en']
                            ev_question = insert_line_breaks(ev_question)
                            ev_q = ev_question.replace(":", '\n')
                            ev_q = insert_line_breaks(ev_q)
                            ev_q = ev_name + '\n\n' + ev_q + '\n \n No.'
                            nodes = [ev_q if ev_question in item else item for item in nodes]
                            antecedents = [ev_q if ev_question in item else item for item in antecedents]
                            symptoms = [ev_q if ev_question in item else item for item in symptoms]
                            rename_mapping = {ev_question: ev_q}
                            graph = nx.relabel_nodes(graph, rename_mapping)
                            color_map[ev_q] = aw.item()
                        else:
                            graph.add_node(value)
                            nodes.append(value)
                            prev_ev = input_vocab[en_in[idx-1].item()]
                            ev_question = evidences[prev_ev]['question_en']
                            ev_name = evidences_with_names.loc[evidences_with_names['Code'] == prev_ev, 'Variable'].values[0]
                            ev_name = ev_name.replace(":", " = ")
                            ev_name = insert_line_breaks(ev_name)
                            connect_to = ev_question.replace(":", ';')
                            connect_to = insert_line_breaks(connect_to)
                            connect_to = ev_name + '\n\n' + connect_to
                            color_map[ev_question] = aw.item()
                            graph.add_edge(connect_to, value)
                            color_map[value] = aw.item()
                    else:
                        is_antecedent = evidences[ev]['is_antecedent']
                        ev_question = evidences[ev]['question_en']
                        ev_name = evidences_with_names.loc[evidences_with_names['Code'] == ev, 'Variable'].values[0]
                        ev_name = ev_name.replace(":", " = ")
                        ev_name = insert_line_breaks(ev_name)
                        ev_question = ev_question.replace(":", ';')
                        ev_question = insert_line_breaks(ev_question)
                        ev_question = ev_name + '\n\n' + ev_question
                        ev_type = evidences[ev]['data_type']
                        if ev_type == 'B':
                            ev_question += '\n \n Yes.'
                        color_map[ev_question] = aw.item()
                        if is_antecedent:
                            antecedents.append(ev_question)
                            nodes.append(ev_question)
                        else:
                            symptoms.append(ev_question)
                            nodes.append(ev_question)

    nx_graph = nx.drawing.nx_pydot.to_pydot(graph)
    dot = gv.Digraph()
    for node in nx_graph.get_nodes():
        node_name = node.get_name().strip('"')
        if node_name not in ('graph', 'node', 'edge'):
            dot.node(node_name, fontsize='42')

    for edge in nx_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        dot.edge(src, dst, penwidth='2')

    ddxs_edge_drawn = []        # there are duplicates in the list, keep track in order to avoid drawing multiple edges between the same nodes
    for node in antecedents:
        if node not in ddxs_edge_drawn:
            dot.edge(node, pathology, penwidth='2')
            ddxs_edge_drawn.append(node)

    for node in symptoms:
        if node not in ddxs_edge_drawn:
            dot.edge(pathology, node, penwidth='2')
            ddxs_edge_drawn.append(node)

    dot.attr(ranksep='3')  # higher distance between nodes, which keeps edges from overlapping (mostly)
    dot.attr(splines='true')  # use splines, so edges won't go through nodes

    norm = mcolors.Normalize(vmin=0, vmax=1)

    cmap = plt.cm.Blues
    for node in nodes:
        node_name = node
        if node_name in color_map:
            color_value = cmap(norm(color_map[node_name]))
            color_hex = mcolors.to_hex(color_value)
            brightness = get_brightness(color_value)
            if brightness < 0.4:
                fontcolor = 'white'
            else:
                fontcolor = 'black'
            dot.node(node_name, style='filled', fillcolor=color_hex, fontcolor=fontcolor, penwidth='2', fontsize='42')
        if node_name == pathology:
            dot.node(node_name, penwidth='5', fontsize='42')
    dot.node(pathology, penwidth='2', fontsize='42')
    gv_graph = dot

    description_file = os.path.join(cfg.EX_PATH, f"{counter}.txt")
    with open(description_file, 'r') as f:
        description_text = f.read()
        f.close()

    gv_graph.attr(xlabel=description_text, labelloc='b', fontsize='12', align='left')

    if cfg.DE_ATTENTION_WEIGHTS_TYPE == 0:
        save_path = os.path.join(cfg.PLOTS_PATH, 'sum', 'pathology_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 1:
        save_path = os.path.join(cfg.PLOTS_PATH, 'mean', 'pathology_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 2:
        save_path = os.path.join(cfg.PLOTS_PATH, 'max', 'pathology_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    elif cfg.DE_ATTENTION_WEIGHTS_TYPE == 3:
        save_path = os.path.join(cfg.PLOTS_PATH, 'full', 'pathology_graphs')
        gv_graph.render(filename=str(counter), directory=save_path, format='png', cleanup=True)
    else:
        print('Please set ATTENTION_WEIGHTS_TYPE variable in config.py to either 0, 1 or 2.')


def assemble_chart(id, title, pathology_name, ddx_names, patient_path):
    pn.extension(comms='vscode')
    css = """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            font-size: 12px;
            line-height: 1.5;
            text-align: justify;
        }
        pre, p {
        font-family: 'Arial', sans-serif;
        font-size: 12px;
        line-height: 1.5;
        text-align: left;
        }
    .card {
        display: flex;
        flex-direction: column;
        border: 1px solid #ccc;
        border-radius: 8px;
        box-shadow: 2px 2px 12px rgba(0, 4, 3, 0.5);
        padding: 16px;
        margin: 8px;
        flex: 1 1 300px;
    }
    .diagnosis_card {
        display: flex;
        flex-direction: column;
        border: 7px solid rgba(108, 108, 125, 1);
        border-radius: 8px;
        box-shadow: 4px 4px 12px rgba(0, 4, 3, 0.5);
        padding: 16px;
        margin: 8px;
        flex: 1 1 300px;
    }
    .card-text {
        font-size: 14px;
        white-space: pre-wrap;
        width: calc(100% - 20px);
        //margin: 0 auto;
        margin-bottom: 12px;
        min-height: 450px;
        text-align:left;
    }
    .centered-image {
        margin: auto;
        max-width: 250px;
        width: 100%;
        height: auto;
    }
    </style>
    """
    pn.config.raw_css.append(css)
    explanation_path = os.path.join(cfg.EX_PATH, str(id)+'.txt')
    try:
        with open(os.path.join(cfg.DATA_DIR, 'placeholder.png'), 'rb') as file:
            placeholder = file.read()
            file.close()
    except FileNotFoundError:
        print("Placeholder image not found. Please put a white picture called placeholder.png in the main dataset directory at "+ cfg.DATA_DIR +'.')

    explanation = read_file(explanation_path, 'r', 'Explanation file not found.')
    explanation_split = explanation.split('split_marker')
    pathology_name = pathology_name.replace('/', '')

    margin=(10, 10,0, 10)  # (top, right, bottom, left) margin

    layout = pn.Row(sizing_mode="stretch_width", margin=margin)

    path_name = pathology_name.replace('/', '')
    age_path = os.path.join(cfg.AgeDiagrams, path_name + '.png')
    sex_path = os.path.join(cfg.SexDiagrams, path_name + '.png')

    age_pie_chart = read_file(age_path, 'rb', placeholder)
    sex_pie_chart = read_file(sex_path, 'rb', placeholder)

    sex_distr = '<b>Distribution among sexes in %:</b>'
    age_distr = '<b>Distribution among age groups in %:</b>'
    card = pn.Column(
        pn.pane.Markdown(
            f"<div class='card-text'>{explanation_split[0]}</div>",
        ).servable(),
        pn.pane.Markdown(
            f"<div class='pre'>{sex_distr}</div>",
        ).servable(),
        pn.pane.PNG(sex_pie_chart, sizing_mode='scale_width', css_classes=["centered-image"]),
        pn.pane.Markdown(
            f"<div class='pre'>{age_distr}</div>",
        ).servable(),
        pn.pane.PNG(age_pie_chart, sizing_mode='scale_width', css_classes=["centered-image"]),
        css_classes=["diagnosis_card"],
    )

    layout.append(card)

    skip_idx = False
    for idx, name in enumerate(ddx_names):
        if name != pathology_name and name not in ['<pad>', '<bos>', '<eos>']:
            if skip_idx:
                text = explanation_split[idx]
            else:
                text = explanation_split[idx + 1]

            ddx_name = name.replace('/', '')
            age_path = os.path.join(cfg.AgeDiagrams, ddx_name + '.png')
            sex_path = os.path.join(cfg.SexDiagrams, ddx_name + '.png')

            age_pie_chart = read_file(age_path, 'rb', placeholder)
            sex_pie_chart = read_file(sex_path, 'rb', placeholder)
            card = pn.Column(
                pn.pane.Markdown(
                    f"<div class='card-text'>{text}</div>",
                ).servable(),
                pn.pane.Markdown(
                    f"<div class='pre'>{sex_distr}</div>",
                ).servable(),
                pn.pane.PNG(sex_pie_chart, sizing_mode='scale_width', css_classes=["centered-image"]),
                pn.pane.Markdown(
                    f"<div class='pre'>{age_distr}</div>",
                ).servable(),
                pn.pane.PNG(age_pie_chart, sizing_mode='scale_width', css_classes=["centered-image"]),
                css_classes=["card"]
            )

            layout.append(card)
        else:
            skip_idx = True

    #layout.show()  # uncomment to show in browser
    save_path = os.path.abspath(os.path.join(cfg.CHART_PATH, str(id)+'.html'))
    layout.save(save_path, embed=True)

