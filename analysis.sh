for i in {0..11}
do
    python imageshower.py test${i}recon/x/49000_x.pth test${i}/x/train True fedhet${i}
done