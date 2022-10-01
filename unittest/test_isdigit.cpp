#include<cstdio>
#include<iostream>
#include<fstream>
#include<string>
using namespace std;
int main(){
    std::fstream file("test2.txt");
    string token, dummy;
    string orient, layer;
    int pointX, pointY;
    char head;
    while(file>>token){
        file >> orient >> layer;
        if (token == "PGON"){
            while ((head = (file >> ws).peek()))
                {
                    // cout << isdigit(head) << endl;
                    if (isdigit(head))
                    {
                        file >> pointX >> pointY;
                        cout << head << " " << pointX << " " << pointY << endl;
                        // polygon->addPoint(pointX, pointY);
                    }
                    else
                        break;
                }
        }
    }
}