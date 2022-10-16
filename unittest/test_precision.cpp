#include<cmath>
#include<iostream>
#include<vector>


int main(){
    float weight = 0.025;
    float MASKRELAX_SIGMOID_STEEPNESS = 4.0;

    float sig_pos = 1/(1+exp(-MASKRELAX_SIGMOID_STEEPNESS * 1));
    float sig_neg = 1/(1+exp(-MASKRELAX_SIGMOID_STEEPNESS * -1));

    float discrete_pos = weight * weight * (-8 * sig_pos + 4);
    float discrete_neg = weight * weight * (-8 * sig_neg + 4);
    std::vector<float> element;
    for(int i = 0; i < 227344; i++){
        element.push_back(discrete_pos);
    }
    for(int i = 0; i < 1411056; i++){
        element.push_back(discrete_neg);
    }
    float sum = 0;
    for(int i = 0; i < 227344 + 1411056; i++){
        sum += element[i];
    }
    std::cout << "by sum " << sum << std::endl;
    std::cout << "by multiply " << 1411056 * discrete_neg + 227344 * discrete_pos << std::endl;
    return 0;
}