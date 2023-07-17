# for i in {0..7}
# do
#     python imageshower.py convfederatedmodbase${i}/x/16000_x.pth convfederated/x/train True 8tri
# done
COEFFS=(21 5 61 1)
MARGINS=(750 1500 2250 3000)
INCREMENT=0
PIDS=(1 2 3 4 5 6 7 8)
for coeff in "${COEFFS[@]}"
do
    for margin in "${MARGINS[@]}"
    do 
        python imageshower.py convactv01l6c${coeff}m${margin}/x/19000_x.pth convtestcont01/x/train False ${coeff}c${margin}m
# done
        
        
    done
    
    
done