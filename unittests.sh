#!/bin/bash

set -o errexit
set -o nounset
set -x

python "modeling/utils/learning_rate_schedule_test.py"
python "modeling/utils/optimization_test.py"
python "modeling/layers/token_to_id_test.py"
python "modeling/layers/fast_rcnn_test.py"
python "readers/vcr_reader_test.py"

echo "DONE"
