# 小妖怪头
SHAPE_ID="003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
# 女人上半身
SHAPE_ID="017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
# 长发男人头
SHAPE_ID="0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

DATA_PATH="${HOME}/chLi/Dataset/pixel_align/${SHAPE_ID}/colmap"

python launch.py \
  --config configs/neus-colmap.yaml \
  --gpu 4 \
  --train \
  dataset.root_dir=${DATA_PATH} \
  dataset.scene=${SHAPE_ID} \
  tag=iter50k \
  seed=0 \
  trainer.max_steps=50000
