import customtkinter as cus
import tkinter as tk
import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

######################################################################################################   

#tạo node của cây
class TreeNode:
    def __init__(self, data,output):
        #lưu data của node
        self.data = data
        #node con
        self.children = {}
        #giá trị đầu ra
        self.output = output
        #vị trí/độ sâu
        self.index = -1
    #thêm node con
    def add_child(self,feature_value,obj):
        self.children[feature_value] = obj

# tính error
def error(y, y_pred):
    sum = 0
    for i in range(0, len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y)  # tra ve trung binh

#thuật toán cây quyết định
class DecisionTreeClassifier:
    def __init__(self):
        # gán nút gốc bằng rỗng
        self.__root = None

    #xác định tần suất xuất hiện của các label
    def count_unique(self,Y):
        d = {}
        for i in Y:
            if i not in d:
                d[i]=1
            else:
                d[i]+=1
        return d
    
    #tính entropy dựa trên tần suất xuất hiện của labels 
    # H(S)=−∑(​pi)*​log2(pi)
    def entropy(self,Y):
        freq_map = self.count_unique(Y)
        entropy = 0
        total = len(Y)
        for i in freq_map:
            p = freq_map[i]/total
            entropy += (-p)*math.log2(p)
        return entropy
    
    #tính IG cho một đặc trưng x
    def information_gain(self,X,Y,selected_feature):
        #tính entropy ban đầu của Y
        info_orig = self.entropy(Y)
        info_f = 0
        #tạo tập hợp các giá trị khác nhau của đặc trưng x
        values = set(X[:,selected_feature])
        df = pd.DataFrame(X)
        #thêm nhãn dự đoán vào cột cuối
        df[df.shape[1]] = Y
        #lấy số lượng tập mẫu trong data ban đầu    
        initial_size = df.shape[0]
        for i in values:
            #lấy giá trị mà giá trị tại cột x có giá trị = i
            df1 = df[df[selected_feature] == i]
            #lấy số lượng mẫu
            current_size = df1.shape[0]
            #tính H(x,S)
            info_f += (current_size/initial_size)*self.entropy(df1[df1.shape[1]-1])
        info_gain = info_orig - info_f #H(S) - H(x,S)
        return info_gain
    
    #cây quyết định
    def decision_tree(self,X,Y,features,level,metric,classes):
        
        # nếu tất cả các mẫu Y tại nút đều cùng 1 lớp thì không chia
        if len(set(Y)) == 1:
            output = None
            for i in classes:
                if i in Y:
                    output = i
            return TreeNode(None,output)

        #nếu không còn đặc trưng để xem xét, trả lại treeNode với output là là lớp có mẫu xuất hiện nhiều nhất
        if len(features) == 0:
            freq_map = self.count_unique(Y)
            output = None
            max_count = -math.inf
            #duyệt qua từng lớp
            for i in classes:
                if i in freq_map :
                    #nếu mẫu hiện tại có tần suất xuất hiện nhiều hơn mẫu trước đó thì thay thế
                    if freq_map[i] > max_count :
                        output = i
                        max_count = freq_map[i]
            return TreeNode(None,output)

        #Tìm đặc trưng tốt nhất để phân chia data tiếp theo
        max_gain = -math.inf
        final_feature = None
        for f in features :
            #tìm đặc trưng có IG lớn nhất
            curr_gain = self.information_gain(X,Y,f)  
            if curr_gain > max_gain :
                max_gain = curr_gain
                #đặc trưng có IG tốt nhất
                final_feature = f
        #gán nhãn cho nút dựa vào tần suất xuất hiện nhiều nhất
        freq_map = self.count_unique(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i in freq_map :
                if freq_map[i] > max_count :
                    output = i
                    max_count = freq_map[i]

            
        unique_values = set(X[:,final_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        #tạo nút hiện tại của cây quyết định. Chọn final_feature là đặc trưng để phân chia tại nút này
        current_node = TreeNode(final_feature,output)

        #loại bỏ đặc trưng đã sử dụng khỏi list đặc trưng
        index  = features.index(final_feature)
        features.remove(final_feature)
        for i in unique_values:
            # tạo data con để tiến hành phân nhóm đệ quy
            df1 = df[df[final_feature] == i]
            # gọi đệ quy để tạo cây con
            node = self.decision_tree(df1.iloc[:,0:df1.shape[1]-1].values,df1.iloc[:,df1.shape[1]-1].values,features,level+1,metric,classes)
            #thêm nút con
            current_node.add_child(i,node)

        #khôi phục đặc trưng đã loại bỏ để sử dụng cho các nhánh khác  
        features.insert(index,final_feature)

        return current_node
    #train mô hình
    def fit(self,X,Y):
        #tạo danh sách các chỉ số đặc trưng dựa trên số lượng đặc trưng trong data
        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0 #cấp ban đầu của cây
        #xây dựng cây
        self.__root = self.decision_tree(X,Y,features,level,"information_gain",classes)

    #gọi đệ quy cho đến khi nó đạt đến một nút lá và có thể trả về một dự đoán
    def __predict_for(self,data,node):      
        # kiểm tra xem nếu đến nút lá rồi thì ngừng, không tìm kiếm thêm nữa
        if len(node.children) == 0 :
            return node.output #trả về giá trị dự đoán tại nút lá
        #lấy giá trị đặc trưng của điểm dữ liệu
        val = data[node.data]
        # nếu đặc trưng val không khớp với bất kì nút con nào, trả về giá trị dự đoán tại nút hiện tại
        if val not in node.children :
            return node.output
        
        # đệ quy để tiếp tục dự đoán
        return self.__predict_for(data,node.children[val])

    #dự đoán nhãn của y
    def predict(self,X):
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.__predict_for(X[i],self.__root)
        return Y

######################################################################################################   

#đọc data
data = pd.read_csv("data.csv")
le = preprocessing.LabelEncoder()

#mã hóa cột diagnosis sang dạng số
data["diagnosis"] = le.fit_transform(data["diagnosis"])

#lấy data cho X và y
X = data.iloc[:,2:]
Y = data.iloc[:,1]
min1, min2 = 999999, 999999
k = 7
kf = KFold(n_splits=k, random_state=None)
for train_index, validation_index in kf.split(data):
    X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
    y_train, y_validation = Y.iloc[train_index], Y.iloc[validation_index]
    #id3
    id3_clf = DecisionTreeClassifier()
    id3_clf.fit(X_train.values,y_train.values)
    #cart
    cart_clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=3, min_samples_leaf=2)
    cart_clf.fit(X_train.values,y_train.values)
    #dự đoán
    y_train_pred1 = id3_clf.predict(np.array(X_train))
    y_validation_pred1 = id3_clf.predict(np.array(X_validation))

    y_train_pred2 = cart_clf.predict(np.array(X_train))
    y_validation_pred2 = cart_clf.predict(np.array(X_validation))

    id3_sum_error = error(np.array(y_train), y_train_pred1) + error(np.array(y_validation), y_validation_pred1)
    cart_sum_error = error(np.array(y_train), y_train_pred2) + error(np.array(y_validation), y_validation_pred2)
    if id3_sum_error < min1:
        min1 = id3_sum_error
        id3_clf1_accuracy_score = accuracy_score(y_validation, y_validation_pred1)
        id3_clf1_precision_score = precision_score(y_validation, y_validation_pred1)
        id3_clf1_recall_score = recall_score(y_validation, y_validation_pred1)
        id3_clf1_f1_score = f1_score(y_validation, y_validation_pred1)
        id3_clf1 = id3_clf

    if cart_sum_error < min2:
        min2 = cart_sum_error
        cart_clf1_accuracy_score = accuracy_score(y_validation, y_validation_pred2)
        cart_clf1_precision_score = precision_score(y_validation, y_validation_pred2)
        cart_clf1_recall_score = recall_score(y_validation, y_validation_pred2)
        cart_clf1_f1_score = f1_score(y_validation, y_validation_pred2)
        cart_clf1 = cart_clf

######################################################################################################   

root = cus.CTk()

# đặt tiêu đề cho chương trình
root.title("Phần mềm chuẩn đoán ung thư vú")

# Tính toán tọa độ trung tâm của màn hình
width = 1200
height = 730
x = (root.winfo_screenwidth() - width) / 1.5
y = (root.winfo_screenheight() - height)/3
# Đặt kích thước và đặt tọa độ cửa sổ ở giữa màn hình
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
#tiêu đề
appTitle = cus.CTkLabel(root,text="Phần mềm chuẩn đoán ung thư vú", font=("Time New Roman", 18, "bold"), text_color="#32d5fa")
appTitle.pack(pady=10)

#tạo frame
biggerFrame = cus.CTkFrame(root, border_width=5)
biggerFrame.pack()
frame = cus.CTkFrame(biggerFrame)
frame.pack(padx=10, pady=10)

#hiển thị độ tin cậy của mô hình dự đoán
infoText1 = cus.CTkLabel(frame, text=f"Mô hình tốt nhất: Cart\nAccuracy: {cart_clf1_accuracy_score}\nPrecision: {cart_clf1_precision_score}\nRecall: {cart_clf1_recall_score}\nF1: {cart_clf1_f1_score}", font=("Time New Roman", 14, "bold"), justify="left")
infoText1.grid(sticky="w", column=0, rowspan=10, padx=10)
infoText2 = cus.CTkLabel(frame, text=f"Mô hình id3(code tay)\nAccuracy: {id3_clf1_accuracy_score}\nPrecision: {id3_clf1_precision_score}\nRecall: {id3_clf1_recall_score}\nF1: {id3_clf1_f1_score}", font=("Time New Roman", 14, "bold"), justify="left")
infoText2.grid(sticky="w", column=0, rowspan=10, padx=10)

values = {}
label_text = [['Radius mean', 'Texture mean', 'Perimeter mean', 'Area mean', 'Smoothness mean', 'Compactness mean', 'Concavity mean', 'Concave points mean', 'Symmetry mean', 'Fractal dimension mean'],
              ['Radius se', 'Texture se', 'Perimeter se', 'Area se', 'Smoothness se', 'Compactness se', 'Concavity se', 'Concave points se', 'Symmetry se', 'Fractal dimension se'],
              ['Radius worst', 'Texture worst', 'Perimeter worst', 'Area worst', 'Smoothness worst', 'Compactness worst', 'Concavity worst', 'Concave points worst', 'Symmetry worst', 'Fractal dimension worst']]

test_value = [[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871],[1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193],[25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]]

#tạo frame thông tin
for i in range(1,4):
    tmp = 0
    for j in range(0,19, 2):
        textLabel = cus.CTkLabel(frame, text=label_text[i-1][j-tmp], font=("Time New Roman", 13, "bold"))
        textLabel.grid(sticky="w", row=j, column=i, padx=40)
        values[f"{label_text[i-1][j-tmp]}"] = cus.CTkEntry(frame, width=90)
        #thêm data tự động
        values[f"{label_text[i-1][j-tmp]}"].insert(0,test_value[i-1][j-tmp])

        values[f"{label_text[i-1][j-tmp]}"].grid(sticky="w", row=j+1, column=i, padx=40, pady=4)
        tmp+=1

#hàm dự đoán
def predict():
    inputValues = [[float(i.get()) for i in values.values()]]

    cart_predictValue = le.inverse_transform(cart_clf1.predict(inputValues))
    id3_predictValue = le.inverse_transform(id3_clf1.predict(inputValues))

    #gán dự đoán vào phần kết quả
    predictText.configure(text="Kết quả dự đoán cart: " + str(cart_predictValue)+"\nKết quả dự đoán id3: " + str(id3_predictValue), justify="left")

#Nút dự đoán và hiển thị kết quả
predictBtn = cus.CTkButton(frame, text="Dự đoán", command=predict)
predictBtn.grid(column=4, row=9, padx=10)
#Hiển thị kết quả
predictText = cus.CTkLabel(frame, text="", font=("Time New Roman", 14, "bold"))
predictText.grid(column=4, row=10, pady=5, padx=5)

root.mainloop()
