#include<set>
#include<iostream>
using namespace std;
int main(){
    std::set<int> vLevel;
    vLevel.insert(3);
    vLevel.insert(1);
    vLevel.insert(2);
    for (set<int>::iterator vIt = vLevel.begin(), vNext = ++vLevel.begin();
         vNext != vLevel.end(); ++vIt, ++vNext){
            std::cout << *vIt << *vNext << std::endl;
         }
}