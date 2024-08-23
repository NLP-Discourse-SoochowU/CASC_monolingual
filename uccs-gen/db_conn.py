import pymysql
import pandas as pd
import os, sys
from datetime import datetime
from sqlalchemy import create_engine
sys.path.append("..")
from app_cfg import *


class dsta_db:
    def __init__(self, host, username, passwd, port, db_name):
        self.host = host
        self.user = username
        self.passwd = passwd
        self.port = port
        self.db_name = db_name
        self.engine = None
        self.connection = None
        self.mycursor = None

    def sqlalchemy_connect(self):
        self.engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(user=self.user, pw=self.passwd, host=self.host, db=self.db_name))

    def sqlalchemy_disconnect(self):
        self.engine.dispose()

    def mysql_connect(self):
        self.connection = pymysql.connect(host=self.host, user=self.user, password=self.passwd, port=self.port, database=self.db_name)
        self.mycursor = self.connection.cursor()

    def mysql_disconnect(self):
        self.mycursor.close()
        self.connection.close()

    def view_tables(self):
        # command = "DESCRIBE comment_clusters"
        # command = "DESCRIBE comment_spans"
        # command = "DESCRIBE events"
        command = "DESCRIBE comments"
        # command = "DESCRIBE news"
        self.mycursor.execute(command)
        print(self.mycursor.fetchall())

    def view_table_data(self):
        # command = "select * from events;"
        # command = "select * from comment_spans"
        # command = "select id, summary, summary_date from comment_clusters where event_id = 8;"
        # command = "select * from comment_clusters"
        # command = "select * from news where article_id = 100;"
        command = "select id from comments where lang = 'BM';"
        self.mycursor.execute(command)
        print(self.mycursor.fetchall())

    def clear_tables(self):
        command = "Delete FROM comment_clusters"
        self.mycursor.execute(command)
        self.connection.commit()

    def test_tables(self):
        # creat event
        sql = "INSERT INTO events (summary) VALUES (%s)"
        val = ("A test data by Longyin, 9/18/2023, with comments")
        self.mycursor.execute(sql, val)
        self.connection.commit()
        comment_cluster_id = self.mycursor.lastrowid

        # view articles and set 100 and 300 as the same event
        sql = "UPDATE news SET event_id = %s WHERE article_id = %s"
        val = (comment_cluster_id, 320180)
        self.mycursor.execute(sql, val)
        self.connection.commit()

        sql = "UPDATE news SET event_id = %s WHERE article_id = %s"
        val = (comment_cluster_id, 320181)
        self.mycursor.execute(sql, val)
        self.connection.commit()

    def get_tuning_documents(self, lang="EN"):
        """ Choose the large amount of articles for model tuning.
        """
        df = None
        try:
            self.mysql_connect()
            query = """select article_id, content from news where lang = '{0}';""".format(lang)
            df = pd.read_sql(query, self.connection)
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Get_tuning_documents error: Waiting for the database to be prepared!")
        return df

    def get_ac_ids(self, ac_id=None):
        """ Return a list of article clusters, where each article is represented by a unique ID.
            Only process those inside the AC but not in the previously obtained CCs.
            [[ID1, ID2], [...], [...]]
        """
        try:
            ac_id_list = list()
            ac_article_id_list = list()
            self.mysql_connect()
            if ac_id is None:
                query = "select id from events where id not in (select event_id from comment_clusters where event_id is not NULL);"
                df = pd.read_sql(query, self.connection)
                for index, row in df.iterrows():
                    ac_id = row[0]
                    ac_id_list.append(ac_id)
                    query_ = "select article_id from news where event_id = {0};".format(ac_id)
                    df_ = pd.read_sql(query_, self.connection)
                    article_list = list()
                    for index_, row_ in df_.iterrows():
                        article_list.append(row_[0])
                    ac_article_id_list.append(article_list)
            else:
                ac_id_list.append(ac_id)
                query_ = "select article_id from news where event_id = {0};".format(ac_id)
                df_ = pd.read_sql(query_, self.connection)
                article_list = list()
                for index_, row_ in df_.iterrows():
                    article_list.append(row_[0])
                ac_article_id_list.append(article_list)
            self.mysql_disconnect()
            return ac_article_id_list, ac_id_list
        except:
            self.mysql_disconnect()
            print("Get_ac_ids error: Waiting for the database to be prepared!")
            return None, None

    def get_comments_by_ids(self, comment_ids):
        try:
            comments = list()
            self.mysql_connect()
            for cmt_id in comment_ids:
                query = "select id, cmt_content, cmt_org_content, lang from comments where id = {0};".format(cmt_id)
                df = pd.read_sql(query, self.connection)
                for index, row in df.iterrows():
                    comments.append((row[0], row[1], row[2], row[3]))
            self.mysql_disconnect()
            return comments
        except:
            self.mysql_disconnect()
            print("Get_comments_by_article_ids error: Waiting for the database to be prepared!")
            return None

    def get_comments_by_article_ids(self, article_cluster):
        """ Return a list of comments.
            (id, cmt_content)
        """
        try:
            comments = list()
            self.mysql_connect()
            for article_id in article_cluster:
                query = "select id, cmt_content from comments where cmt_article_id = {0};".format(article_id)
                df = pd.read_sql(query, self.connection)
                for index, row in df.iterrows():
                    comments.append((row[0], row[1]))
            self.mysql_disconnect()
            return comments
        except:
            self.mysql_disconnect()
            print("Get_comments_by_article_ids error: Waiting for the database to be prepared!")
            return None

    def get_comment_trees_by_article_ids(self, article_cluster):
        """ Return a list of comments.
            (id, cmt_content, cmt_tree)
        """
        try:
            comments = list()
            self.mysql_connect()
            for article_id in article_cluster:
                query = "select id, cmt_content, cmt_tree from comments where cmt_article_id = {0};".format(article_id)
                df = pd.read_sql(query, self.connection)
                for index, row in df.iterrows():
                    comments.append((row[0], row[1], row[2]))
            self.mysql_disconnect()
            return comments
        except:
            self.mysql_disconnect()
            print("Get_comment_trees_by_article_ids error: Waiting for the database to be prepared!")
            return None

    def get_article_by_ids(self, article_ids):
        """ Return article info for the given article ID.
        """
        try:
            self.mysql_connect()
            articles = list()
            for article_id in article_ids:
                query = "select title, content from news where article_id = {0};".format(article_id)
                df = pd.read_sql(query, self.connection)
                for index, row in df.iterrows():
                    articles.append(tuple(row))
            self.mysql_disconnect()
            return articles
        except:
            self.mysql_disconnect()
            print("Get_article_by_id error: Waiting for the database to be prepared!")
            return None


    def get_uccs_by_ac_id(self, ac_id):
        """ Based on the article_cluster ID, search for uccs information.
            from ac id get CCs directly; (('id', 'int'), ('summary', 'text'), ('summary_date', 'datetime'), ('event_id', 'int'))  CC
            from CC id get semantic spans. (id, int), (cmt_id, int), (comment_cluster_id, int), (semantic_span_content, text)
        """
        comment_clusters = list()
        try:
            self.mysql_connect()
            # 1. check if ac_id exists in the article cluster Table
            query_check = "select * from events where id = {0};".format(ac_id)
            df = pd.read_sql(query_check, self.connection)
            if df.empty:
                self.mysql_disconnect()
                return None, 402

            # 2. get cc_ids from cache
            query = "select id, summary, summary_date from comment_clusters where event_id = {0};".format(ac_id)
            df = pd.read_sql(query, self.connection)
            if df.empty:
                self.mysql_disconnect()
                return None, 404

            for index, row in df.iterrows():
                cc_id, summary_txt = row[0], row[1]
                query_ = "select id, cmt_content from comments where cmt_cluster_id = {0};".format(cc_id)
                df_ = pd.read_sql(query_, self.connection)
                comment_info = list()
                if df_.empty:
                    self.mysql_disconnect()
                    return None, 404
                for index_, row_ in df_.iterrows():
                    comment_id, comment_content = row_[0], row_[1]
                    comment_info.append({"comment_id": comment_id, "comment_content": comment_content})
                comment_clusters.append({"comment_cluster_id": cc_id, "comments": comment_info, "summary": summary_txt})
            return_body = {"article_cluster_id": ac_id, "comment_clusters": comment_clusters}
            self.mysql_disconnect()
            return return_body, 200
        except:
            self.mysql_disconnect()
            print("Get_uccs_by_ac_id: Database Error.")
            return None, 401

    def delete_comment_clusters(self, ac_id):
        """ Delete all cc by ac_id, because one ac_id determine a specific group of ccs.
        """
        try:
            self.mysql_connect()
            sql_ = "Delete FROM comment_clusters where event_id = %s"
            self.mycursor.execute(sql_, ac_id)
            self.connection.commit()
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Delete_comment_clusters error: Waiting for the database to be prepared!")

    def write_comment_cluster(self, summary_all, ac_id):
        """ Return a comment cluster ID.
            (('id', 'int'), ('summary', 'text'), ('summary_date', 'datetime'), ('event_id', 'int'))  CC
            Also update the comments table.
        """
        sum_date = datetime.now()
        comment_cluster_id = None
        try:
            self.mysql_connect()
            sql = "INSERT INTO comment_clusters (summary, summary_date, event_id) VALUES (%s, %s, %s)"
            val = (summary_all, sum_date, ac_id)
            self.mycursor.execute(sql, val)
            self.connection.commit()
            comment_cluster_id = self.mycursor.lastrowid
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Write_comment_cluster error: Waiting for the database to be prepared!")
        return comment_cluster_id

    def update_comment_table(self, cmt_id_all, cc_id):
        try:
            self.mysql_connect()
            for cmt_id in cmt_id_all:
                sql = "UPDATE comments SET cmt_cluster_id = %s WHERE id = %s"
                val = (cc_id, cmt_id)
                self.mycursor.execute(sql, val)
                self.connection.commit()
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Write_comment_table error: Waiting for the database to be prepared!")

    def delete_comment_spans(self, cc_id):
        """ Delete those with None cc_id, and current cc_id
        """
        try:
            self.mysql_connect()
            sql_ = "Delete FROM comment_spans WHERE comment_id = %s OR comment_id IS NULL"
            self.mycursor.execute(sql_, cc_id)
            self.connection.commit()
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Delete_comment_spans error: Waiting for the database to be prepared!")

    def write_comment_spans(self, cmt_id_all, cc_id, semantic_span_content_all):
        """ (id, int), (cmt_id, int), (comment_cluster_id, int), (semantic_span_content, text)
        """
        try:
            self.mysql_connect()
            for cmt_id, semantic_span_content in zip(cmt_id_all, semantic_span_content_all):
                sql = "INSERT INTO comment_spans (comment_cluster_id, content, comment_id) VALUES (%s, %s, %s)"
                val = (cc_id, semantic_span_content, cmt_id)
                self.mycursor.execute(sql, val)
                self.connection.commit()
            self.mysql_disconnect()
        except:
            self.mysql_disconnect()
            print("Write_comment_spans error: Waiting for the database to be prepared!")

if __name__ == "__main__":
    pass
