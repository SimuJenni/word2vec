DATADIR="word2vec_data"

if [ ! -d "$DATADIR" ]; then
  mkdir $DATADIR
fi

if [ ! -d "$DATADIR/tmp" ]; then
  mkdir "$DATADIR/tmp"
fi

if [ ! -d "$DATADIR/results" ]; then
  mkdir "$DATADIR/results"
fi

if [ ! -d "$DATADIR/eval” ]; then
  mkdir "$DATADIR/eval”
fi

if [ ! -e "$DATADIR/text8" ]; then
  wget http://mattmahoney.net/dc/text8.zip -O "$DATADIR/text8.zip"
  unzip "$DATADIR/text8.zip" -d $DATADIR
fi

wget http://word2vec.googlecode.com/svn/trunk/questions-words.txt -O "$DATADIR/questions-words.txt"
