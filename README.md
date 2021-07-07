# DL_HW1

成大測量 深度學習作業一'

python environment numpy,matplotlib,tensorflow

keras database fashion minst

這次實習是要分析三種不同深度的網路模型分類精準度，而我選擇的三個深度分別是一層、十層、三十層。
分類的資料來源為 fashion_mnist，並且總共分為 10 類。使用線性的 NN model。
1.  三十層

![image](https://user-images.githubusercontent.com/61956056/124768977-af459600-df6b-11eb-9f3a-f9aa6410a64c.png)![image](https://user-images.githubusercontent.com/61956056/124769000-b2d91d00-df6b-11eb-9504-7740c3fa6867.png)

以上為30層epochs為10的train history，從圖的趨勢可以看出訓練正確率還在上升，代表還未收斂。而其Accuracy of testing data = 27.4%，正確率頗低。從兩條線的擬合程度可以看出訓練成果並沒有overfitting。

![image](https://user-images.githubusercontent.com/61956056/124769133-cedcbe80-df6b-11eb-970c-ab3e55836a9f.png)![image](https://user-images.githubusercontent.com/61956056/124769160-d2704580-df6b-11eb-8678-7c0b5125705f.png)

以上為30層epochs為30的train history，從圖的趨勢可以看出訓練正確率還在上升，代表還未收斂，而其Accuracy of testing data = 70.9%，上升速度頗快，但validation的線條不斷震盪，代表模型訓練成果還不穩定。

![image](https://user-images.githubusercontent.com/61956056/124769288-ee73e700-df6b-11eb-9072-0fcb94949f7a.png)![image](https://user-images.githubusercontent.com/61956056/124769305-f16ed780-df6b-11eb-98da-c4dd1e72d69d.png)

以上為30層epochs為100的train history，從圖的趨勢可以看出來有趨於平緩，而其Accuracy of testing data = 80.6%，正確率也趨於穩定。從validation的線條可以看出兩線條在後期擬合得不錯，代表訓練成果沒有overfitting。

2.  十層

![image](https://user-images.githubusercontent.com/61956056/124769384-021f4d80-df6c-11eb-979a-ef2e2148ed93.png)![image](https://user-images.githubusercontent.com/61956056/124769408-064b6b00-df6c-11eb-8846-e85239165728.png)

以上為10層epochs為10的train history，從圖的趨勢可以看出訓練正確率不斷上升但趨於平緩，有收斂的可能。而其Accuracy of testing data = 78.3%，從兩條線的擬合程度可以看出訓練成果並沒有overfitting。

![image](https://user-images.githubusercontent.com/61956056/124769465-0ea3a600-df6c-11eb-8ac0-f4a02cca0088.png)![image](https://user-images.githubusercontent.com/61956056/124769482-119e9680-df6c-11eb-99e6-19a503222bf0.png)

以上為10層epochs為20的train history，從圖的趨勢可以看出訓練成果已呈現平緩，代表還未收斂，從validation的線條可以看出兩線條在後期有發散的趨勢，代表訓練成果可能有overfitting。而其Accuracy of testing data = 81.9%，相較model內部的準確度低，代表可能有local minima。

![image](https://user-images.githubusercontent.com/61956056/124769524-1c592b80-df6c-11eb-8d6f-669782a11c7c.png)![image](https://user-images.githubusercontent.com/61956056/124769541-20854900-df6c-11eb-9bea-e4947922509d.png)

以上為10層epochs為30的train history，從圖的趨勢可以看出訓練正確率已平緩，從validation的線條可以看出兩線條在後期已經過擬合，代表訓練成果有overfitting。而其Accuracy of testing data = 76.3%，正確率降低，與model內部的準確度相比差距大，應該有Local minima。

3.  一層

![image](https://user-images.githubusercontent.com/61956056/124769664-3abf2700-df6c-11eb-94d6-00313c1eee2a.png)![image](https://user-images.githubusercontent.com/61956056/124769676-3dba1780-df6c-11eb-8fdb-46c48dc5286b.png)

以上為1層epochs為10的train history，從圖的趨勢可以看出訓練正確率還在上升，但已趨於平緩，從圖可以看出兩條線還未重合，代表尚未擬合還有訓練的空間。而其Accuracy of testing data = 78.1%。

![image](https://user-images.githubusercontent.com/61956056/124769711-46125280-df6c-11eb-8ec9-5925d120813d.png)![image](https://user-images.githubusercontent.com/61956056/124769730-4b6f9d00-df6c-11eb-88f3-1514623be8e1.png)

以上為1層epochs為100的train history，從圖的趨勢已經平緩，且兩條線高度擬合，大概從epoch 80左右開始有一點過擬合的情形，可能有些許over fitting。而其Accuracy of testing data = 81.9%，正確率上升不多。

4.  討論

從前述的結果可以看出訓練成果由三十層=>十層=>一層各自的特性都不盡相同。三十層的準確度震盪較大，可能是因為層數較多當epoch太多時容易提取錯誤的特徵，造成模型不穩定。十層的前期擬合不錯，但到了大概epoch 15的時候出現過擬合的情況。一層的模型一直都蠻穩定，在前期就有收斂的傾向，是個可以快速找到特徵的模型。比較三個模型的test accuracy，三十層一開始準確率低上升速度也不快，最高準確率有到80.6%；十層的正確率還不錯，但後期與train accuracy差距較大，可能有Local minima；一層的正確率一開始就蠻高的，並且直到epoch 100依舊保持在81.9%。由這三個模型的訓練成果比較，可以大致判斷較合適的層數在一層到十層之間。







