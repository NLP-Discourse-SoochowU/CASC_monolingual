"""
Author: Lynx Zhang
Date: 2020.
Email: zzlynx@outlook.com
"""
from nltk.tree import Tree
# from nltk.draw.tree import TreeView
import spacy
from spacy import displacy


class comment_tree:
    def __init__(self, c_id=None, p_id=None, comment_text=None, children=None):
        self.c_id = c_id
        self.p_id = p_id
        self.comment_text = comment_text
        self.children = children

    def read_tree(self):
        print("Father: ", self.comment_text)
        print("Children: ", self.children)
        print("============================")
        for idx, child in enumerate(self.children):
            print("Child: ", idx)
            child.read_tree()

    def draw_tree(self):
        root_str = self.form_string()
        input(root_str)
        # t = Tree.fromstring(root_str)
        # TreeView(t)._cframe.print_to_file('data/drawer/output.ps')

    def form_string(self):
        comment_words = self.comment_text.strip().split()
        insert_id, repeat_id = 6, 1
        result_comment = ""
        for idx, word in enumerate(comment_words):
            if idx == insert_id * repeat_id:
                result_comment += (word + "\\ ")
                repeat_id += 1
            else:
                result_comment += (word + "<>")
        result_comment = result_comment.strip()
        if len(self.children) > 0:
            root_str = "[" + result_comment + " "
            for child in self.children:
                child_str = child.form_string()
                root_str += (child_str + " ")
            root_str = root_str.strip() + "]"
        else:
            root_str = "[" + result_comment + "]"
        return root_str

    def draw_dep(self, depth=1):
        comment_words = self.comment_text.strip().split()
        self_txt = " ".join(comment_words)
        if len(self.children) > 0:
            root_str = "(" + self_txt + " "
            for child in self.children:
                child_str = child.draw_dep(depth + 1)
                root_str += (child_str + " ")
            root_str = root_str.strip() + ")"
        else:
            root_str = self_txt
        return root_str

    def get_tree_comments(self, file_name, tree_id, depth=1):
        comment_words = self.comment_text.strip().split()
        self_txt = [file_name + "$^$" + str(tree_id) + "$^$" + str(depth) + "$^$" + " ".join(comment_words)]
        if len(self.children) > 0:
            for child in self.children:
                child_text = child.get_tree_comments(file_name, tree_id, depth + 1)
                self_txt += child_text
        return self_txt
