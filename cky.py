import nltk
import re
from nltk.corpus import treebank, ptb
from itertools import product, accumulate
import random
import sys
from sklearn.model_selection import train_test_split
nltk.download('treebank')
nltk.download('tagsets')

tree = list(treebank.parsed_sents())
tree, test = train_test_split(tree, test_size = 0.1)

def only_pos(tree):
    to_be_deleted = []
    for i in range(len(tree)):
        if not isinstance(tree[i], nltk.tree.Tree):
            continue
        if not re.search('[a-zA-Z]', tree[i].label()):
            to_be_deleted.append(i)
            
        else:
            tree[i]._label = re.sub('-.*', '', tree[i].label())
            only_pos(tree[i])
            
    for i in sorted(to_be_deleted, reverse = True):
        tree.__delitem__(i)
        
def rule_count(tree, c_label):
    target = ''
    for t in tree:
        if not re.search('[a-zA-Z]', t.label()):
            continue
            
        if isinstance(t, nltk.tree.Tree) and isinstance(t[0], nltk.tree.Tree):
            target += t.label() + ' '
            #rules.append(c_label + ' --> ' + t.label())
            rule_count(t, t.label())
        elif isinstance(t, nltk.tree.Tree):
            target += t.label() + ' '
            #rules.append(c_label + ' --> ' + t.label())
            
    rules.append(c_label + ' --> ' + target)
    
def rule_count_with_words(tree, c_label):
    target = ''
    if isinstance(tree, str):
        #print('I am here')
        rules.append(c_label + ' --> ' + tree)
        return
    
    for t in tree:        
        if isinstance(t, nltk.tree.Tree) and isinstance(t[0], nltk.tree.Tree):
            target += t.label() + ' '
            #rules.append(c_label + ' --> ' + t.label())
            rule_count_with_words(t, t.label())
        elif isinstance(t, nltk.tree.Tree):
            target += t.label() + ' '
            rule_count_with_words(t, t.label())
            #rules.append(c_label + ' --> ' + t.label())
            
        else:
            rule_count_with_words(t, c_label)
    
    if target == '' or target == ' ':
        return
    rules.append(c_label + ' --> ' + target)
	
	
	
# Remove . and , 
x = [only_pos(t) for t in tree]
x = [only_pos(t) for t in test]
# Normal form
x = [nltk.treetransforms.chomsky_normal_form(t) for t in tree]
x = [nltk.treetransforms.chomsky_normal_form(t) for t in test]
# Remove unaries
x = [nltk.treetransforms.collapse_unary(t, collapsePOS = True) for t in tree]
x = [nltk.treetransforms.collapse_unary(t, collapsePOS = True) for t in test]


all_rules = []
for i,t in enumerate(tree):
    rules = []
    try:
        rule_count_with_words(t, 'S')
        
    except:
		pass
        #print(i)
    
    all_rules += [re.sub('[+=0-9]', '', re.sub('-[0-9]+', '', re.sub('[+][^\s]+', '', x))) for x in rules] #Collapse ROOT
	
rule_dict = [x.split(' --> ') for x in all_rules]

leaf_rules = [x for x in all_rules if len(x.split(' --> ')[1].split(' ')) == 1 and x.split(' --> ')[1] != '']
leaf_rules = [' --> '.join(x) for x in [[y.split(' --> ')[0], y.split(' --> ')[1]] for y in leaf_rules]]
cfg_rules = [x for x in all_rules if len(x.split(' --> ')[1].split(' ')) > 1]


leaf_parent = [x.split(' --> ')[0] for x in leaf_rules if len(x.split(' --> ')[0]) != 0]

leaf_uni_prob = nltk.probability.FreqDist(leaf_parent)

denom = float(leaf_uni_prob.N())
for word in leaf_uni_prob:
    leaf_uni_prob[word] /= denom
	
vocab = list(set([x.split(' --> ')[1] for x in leaf_rules]))

leaf_uni_cumu_prob = list()
for k,v in zip(leaf_uni_prob.keys(), accumulate(leaf_uni_prob.values())):
    leaf_uni_cumu_prob.append((k, v))
	
leaf_uni_cumu_prob = sorted(leaf_uni_cumu_prob, key = lambda x:x[1])

def generate_leaf_rule():
    rand = random.random()
    for k, v in leaf_uni_cumu_prob:
        if rand < v:
            return k, leaf_uni_prob[k]
			
leaf_rule_prob = nltk.probability.FreqDist(leaf_rules)
cfg_rule_prob = nltk.probability.FreqDist(cfg_rules)

denom_leaf = sum(leaf_rule_prob.values())
denom_cfg = sum(cfg_rule_prob.values())
for k in leaf_rule_prob.keys():
    leaf_rule_prob[k] /= denom_leaf
    
for k in cfg_rule_prob.keys():
    cfg_rule_prob[k] /= denom_cfg
	
class Rule:
    def __init__(self, rule_type, root, child_1, child_2, probability):
        self.rule_type = rule_type
        self.parent = root
        self.child_1 = child_1
        
        if rule_type == 'NT':
            self.child_2 = child_2
            
        self.prob = probability
		
def print_list(x):
    for a in x:
        print(a)

def mutate_rules(rule_list_1, rule_list_2):
    pairs = list(product(rule_list_1, rule_list_2))
    rules = []
    for c_1, c_2 in pairs:
        #print([(rule, prob) for rule, prob in cfg_rule_prob.items() if ' {} {} '.format(*[c_1.parent, c_2.parent]) in rule])
        dummy = [rules.append(Rule('NT', rule.split(' --> ')[0], c_1, c_2, c_1.prob*c_2.prob*prob)) for rule, prob in cfg_rule_prob.items() if ' {} {} '.format(*[c_1.parent, c_2.parent]) in rule]
        
    return rules
		
def generate_boxes(x):
    num_pairs = abs(x[0] - x[1])
    offsets = [[0, num_pairs], [-1, 0]]
    box_pairs = []
    for i in range(num_pairs):
        box_pairs.append([tuple(sum(y) for y in zip(x, offsets[0])), tuple(sum(y) for y in zip(x, offsets[1]))])
        offsets[0][1] -= 1
        offsets[1][0] -= 1
        
    return box_pairs


def recurse_tree(rule, depth):
    if rule.rule_type == 'T':
        for i in range(depth):
            print('\t', end = '')
        print(rule.parent + ' - ' + rule.child_1)
        return rule.parent
    
    for i in range(depth):
        print('\t', end = '')
        
    print('(' + rule.parent)
    to_be_returned = recurse_tree(rule.child_2, depth + 1) + ' ' + recurse_tree(rule.child_1, depth + 1)
    
    for i in range(depth):
        print('\t', end = '')
    print(')')
    return to_be_returned

def print_tree(rule):
    global str_tree
    if rule.rule_type == 'T':
        str_tree += '(' + rule.parent + ' ' + rule.child_1 + ')'
        return
        
    
    str_tree+='(' + rule.parent + ' '
    print_tree(rule.child_2), print_tree(rule.child_1)
    str_tree+=')'
    

def print_tree(rule):
    global str_tree
    if rule.rule_type == 'T':
        str_tree += rule.parent + ' '
        return
        
    
    print_tree(rule.child_2), print_tree(rule.child_1)
    

def parse_tree(sentence):
    sent_words = sentence.split(' ')
    parse = [[[] for x in range(y + 1)] for y in range(len(sent_words))]

    for i, w in enumerate(sent_words):
        if w not in vocab:
            for i_ in range(3):
                pos, prob_ = generate_leaf_rule()
                parse[i][i] += [Rule('T', pos, w, None, prob_)]
        else:
            parse[i][i] += [Rule('T', rule.split(' --> ')[0], rule.split(' --> ')[1], None, prob) for rule, prob in leaf_rule_prob.items() if rule.split(' --> ')[1] == w]

    for i in range(1, len(sent_words)):
        for j in range(len(sent_words) - i):
            this_box_rules = []
            #print('{}, {}'.format(*[i + j, j]))
            for boxes in generate_boxes([i + j, j]):
                box_1 = boxes[0]
                box_2 = boxes[1]
                this_box_rules += mutate_rules(parse[box_1[0]][box_1[1]], parse[box_2[0]][box_2[1]])
                #print(len(mutate_rules(parse[box_1[0]][box_1[1]], parse[box_2[0]][box_2[1]])))
                try:
                    pass
                    #print(vars(this_box_rules[-1]))

                except:
                    pass
        
            parse[i + j][j] += sorted(this_box_rules, key = lambda x:x.prob)[:5]
            
    global str_tree
    str_tree = ''
    print_tree(parse[-1][0][[x.prob for x in parse[-1][0]].index(max([x.prob for x in parse[-1][0]]))])
    return str_tree#.replace(')(', ') (')

to_parse = ' '.join(sys.argv[1:])
print(parse_tree(to_parse))
