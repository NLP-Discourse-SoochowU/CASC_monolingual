# CASC
Zhang et al., Comprehensive Abstractive Comment Summarization with Dynamic Clustering and Chain of Thought. ACL2024

# Environment
Please refer to the python packages in file "./pkglist.txt"

# Data
Our constructed evaluation dataset can be found in "uccs-gen/data/mysql_db/test_updd.json"

# Evaluation
Follow the `uccs-gen/evaluate.sh` file to test the two stages separately.
Follow the `uccs-gen/evaluate_pipeline.sh` file to test the pipeline system.

```bash
cd uccs-gen
bash evaluate.sh
bash evaluate_pipeline.sh
```

Contact [Longyin Zhang](zhangly@i2r.a-star.edu.sg) for more info.