SEED = 19
BATCH_SIZE = 64
carticle_clusters = "data/mysql_db/article_clusters.csv"
carticle_name2comments = "data/mysql_db/comment_db.csv"
carticle_name2trees = "data/mysql_db/comment_tree_db.csv"
cached_response_path = "data/mysql_db/cached_responses.json"

cached_semantic_group = "data/cache/cached_semantic_group.pkl"
cached_ac_ids = "data/cache/cached_ac_path.pkl"
cached_w2v = "data/cache/w2v_embedding"

output_file = "data/uccs_test/output.summary"
output_tt_file = "data/uccs_test/output.summary_title"

# evaluation
manual_dev_path = "data/mysql_db/dev_updd.json"
manual_test_path = "data/mysql_db/test_updd.json"
eval_summary_path = "data/uccs_test/test.summary"
eval_article_path = "data/uccs_test/test.article"
name2background = "data/mysql_db/name2background.json"
