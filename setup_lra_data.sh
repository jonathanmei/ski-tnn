
# LRA data
#wget https://storage.googleapis.com/long-range-arena/lra_release.gz
#wget http://tangra.cs.yale.edu/newaan/data/releases/2014/aanrelease2014.tar.gz

tar -zxvf lra_release.gz -C ..  # creates ../lra_release
tar -zxvf aanrelease2014.tar.gz -C ../lra_release  # creates ../lra_release/aan
cd ../lra_release
rm -rf listops-1000
rm -rf pathfinder128
mv lra_release/* .
rmdir lra_release
mv listops-1000 listops
mv tsv_data aan
