#include<iostream>
using namespace std;

int main(){
    bool a = 1;
    bool b = 0;
    cout << (!a || b);
    cout << endl;
    cout << (!(a||b));
    cout << endl;
    cout << ((!a) || b);   //正确运算顺序
    cout << endl;
    return 0;
}