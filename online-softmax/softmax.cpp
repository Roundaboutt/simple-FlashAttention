#include<iostream>
#include<vector>
#include<cmath>
#include<random>


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

void init_vector(std::vector<float>& a, std::vector<float>& b){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1, 10);

    for (auto &x : a){
        x = dis(gen);
    }

    for (auto &x : b){
        x = dis(gen);
    }    
}

int main(){
    std::vector<float> src(10);
    std::vector<float> value(10);

    init_vector(src, value);

    float res1 = softmax_with_dotProduct(src, value);
    float res2 = online_softmax_with_dotProduct(src, value);
    if (abs(res1 - res2) <= 1e-5){
        std::cout<< "result match!" << std::endl;
    }
    else{
        std::cout<< "result not match!" << std::endl;
    }
}