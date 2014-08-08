DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

DATA_PATH=$DATA_DIR
VECTOR_DATA=$DATA_DIR/word-vector.bin

echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/WordVec -train $DATA_DIR -output $VECTOR_DATA -hidden_size 200 -window 5 -threads 4 -prefix text8_
  

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
