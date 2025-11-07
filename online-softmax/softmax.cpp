#include<iostream>
#include<vector>
#include<cmath>

float softmax_with_dotProduct(std::vector<float> &src, std::vector<float> &value){
    float res = 0.f;
    float sum = 0.f;
    float max_value = -INFINITY;

    for (int i = 0; i < src.size(); i++){
        max_value = std::max(max_value, src[i]);
    }

    for (int i = 0; i < src.size(); i++){
        sum += std::exp(src[i] - max_value);
    }

    for (int i = 0; i < src.size(); i++){
        res += std::exp(src[i] - max_value) * value[i] / sum;
    }
    return res;
}


float online_softmax_with_dotProduct(std::vector<float> &src, std::vector<float> &value){ 
    float res = 0.f;
    float max_value = -INFINITY;
    float pre_max_value = 0.f;

    float pre_sum = 0.f;
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++){
        max_value = std::max(max_value, src[i]);
        sum = pre_sum * std::exp(pre_max_value - max_value) + std::exp(src[i] - max_value);
        res = res * pre_sum * std::exp(pre_max_value - max_value) / sum + std::exp(src[i] - max_value) * value[i] / sum;

        pre_max_value = max_value;
        pre_sum = sum;
    }
    return res;
}

int main(){
    std::vector<float> src = {1, 2, 3, 4, 5};
    std::vector<float> value = {1, 2, 3, 4, 5};

    float res1 = softmax_with_dotProduct(src, value);
    float res2 = online_softmax_with_dotProduct(src, value);
    std::cout << res1 << std::endl;
    std::cout << res2 << std::endl;
}