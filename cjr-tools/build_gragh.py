import os
from py2neo import Graph, Node
from utils.export_data import export_data
from utils.read_csv import read_csv

class PoemGraph(object):
    """构建知识图谱"""

    def __init__(self):
        super(PoemGraph, self).__init__()
        self.path = os.path.split(os.path.realpath(__file__))[0] + \
                    os.sep + "poemData" + os.sep + "csv" + os.sep + "all.csv"
        self.graph = Graph("http://localhost:7474/", auth=("neo4j", "20230224"))  # 链接至图谱
        self.graph.run("match(n) detach delete(n)")  # 删除现有图谱

    def create_node(self, label, nodes):
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.graph.create(node)

    def create_poem_node(self, poem_info):
        count = 0
        for poem_dict in poem_info:
            node = Node("Poem", name=poem_dict["name"],
                        content=poem_dict["content"])
            count += 1
            self.graph.create(node)
        print(count)

    def create_graph_node(self):
        author, dynasty, tag, name, content, rel_a_d, rel_d_a, rel_a_t, rel_t_a, rel_d_t, rel_t_d, rel_p_t, rel_t_p, \
        rel_p_c, rel_info = read_csv(self.path)
        self.create_poem_node(rel_info)
        self.create_node("Author", author)
        self.create_node("Dynasty", dynasty)
        self.create_node("Tag", tag)

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))

        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.graph.run(query)
            except Exception as e:
                print(e)

    def create_graph_rels(self):
        author, dynasty, tag, name, content, rel_a_d, rel_d_a, rel_a_t, rel_t_a, rel_d_t, rel_t_d, rel_p_t, rel_t_p, \
        rel_p_c, rel_info = read_csv(self.path)
        self.create_relationship("Author", "Dynasty", rel_a_d, "Author_IS_Dynasty", "所属朝代")
        self.create_relationship("Dynasty", "Author", rel_d_a, "Dynasty_THE_Author", "存在")
        self.create_relationship("Author", "Poem", rel_a_t, "Author_THE_Poem", "写作")
        self.create_relationship("Poem", "Author", rel_t_a, "Poem_IS_Author", "作者")
        self.create_relationship("Dynasty", "Poem", rel_d_t, "Dynasty_THE_Poem", "朝代包含诗")
        self.create_relationship("Poem", "Dynasty", rel_t_d, "Poem_IS_Dynasty", "诗属于朝代")
        self.create_relationship("Poem", "Tag", rel_p_t, "Poem_IS_Tag", "诗的标签")
        self.create_relationship("Tag", "Poem", rel_t_p, "Tag_THE_Poem", "标签的诗")

poem = PoemGraph()
poem.create_graph_node()
poem.create_graph_rels()
export_data(poem.path)