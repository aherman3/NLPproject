# NLP Project #  
# Annie Hermann #

### Baseline ###  
Code in baseline.py  
Baseline testing output:    
Total F1: 0.3621471159325782  

### Naive Bayes Classifier ###  
Code in naive_bayes.py  
Naive Bayes Classifier testing output:  
Total F1: 0.2339127632039389  

### Neural Network ###  
Code in neural_net.py  
Neural Net testing output:  
Total F1: 0.403  
Neural Net + NB Classifier testing output:  
Total F1: 0.4304520261023754  

### Demo ###  
Run demo on text with `./demo.py -t (sentence)`  
- try out `./demo.py -t 不管黑猫白猫，捉到老鼠就是好猫` (Deng Xiao-ping's famous quote, "It doesn't matter if it's a black cat or a white cat, as long as it can catch a mouse it's a good cat")  

Run demo on textfile with `./demo.py -f (file)`  
- try out `./demo.py -f data/test/final_project.txt` (my Chinese final project)
