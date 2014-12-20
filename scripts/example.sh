DIR=`pwd`
cd $DIR

DATA_DIR=../data
BIN_DIR=../bin
VECTOR_DATA=$DATA_DIR/word-vector.bin


echo -- Training vectors...
time $BIN_DIR/WordVec -train $DATA_DIR -output $VECTOR_DATA -hidden_size 200 -window 5 -iter 2 -threads 4 -prefix text8_
  

echo -- calculate word distance...

$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
