#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

float data [20][5];
void read();
void error(double theta[5]);
void normalize();
void gradient(double theta[5], double alpha);
void runner(double theta[5], double alpha, int interations);

int main(int argc, char *argv[]){
    read();
    double theta[5] = {0, 0, 0, 0, 0};
    double finality[5];//holds answer
    string classify[5];//holds good bad and ugly
    double alpha = .001;
    int iterations = 1000;
	normalize();//normalize data
	runner(theta, alpha, iterations);//start program
	cout << "After " << iterations << " iterations. We get the following thetas: " << endl;
	for(int i = 0; i < 5; i++){
        cout << theta[i] << endl;
	}
	cout << "Predicting: " << endl;
	for(int j = 0; j < 5; j++){
        finality[j] = theta[0];//gives finality an inital theta
	}
    for(int i = 1; i < 5; i++){//calculate predicted answer
        finality[0] = finality[0] + theta[i]*data[15][i];
        finality[1] = finality[1] + theta[i]*data[16][i];
        finality[2] = finality[2] + theta[i]*data[17][i];
        finality[3] = finality[3] + theta[i]*data[18][i];
        finality[4] = finality[4] + theta[i]*data[19][i];
    }
    for(int q = 0; q < 5; q++){
        if(finality[q] < 125)
            classify[q] = "bad";
        else if(finality[q] < 275)
            classify[q] = "ok";
        else
            classify[q] = "good";
    }
    cout << "I predict that the 5 reviewers will have a review score of: " << endl
         << finality[0] << " " << classify[0] << endl
         << finality[1] << " " << classify[1] << endl
         << finality[2] << " " << classify[2] << endl
         << finality[3] << " " << classify[3] << endl
         << finality[4] << " " << classify[4] << endl;
    error(theta);
	return 0;
}

void read(){
    ifstream file;//all 20 entries will be in one list first 15 is training
    file.open("data.txt");
    float temp = 0;
    int i, j = 0;
    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 5; j++){
                file >> temp;
                data[i][j] = temp;
        }
    }
    file.close();
}

void error(double theta[5]){//Done?
    double total = 0;
    for(int i = 0; i < 20; i++){//for each data point
        total = total + pow(data[i][0] - (theta[1]*data[i][1] + theta[2]*data[i][2] + theta[3]*data[i][3] +
                                          theta[4]*data[i][4] + theta[0]), 2);
    }
    total = total / 20;
    cout << "The total error rate for all data (testing and training) is " << total << endl;
}

void normalize(){//updated
    for(int j = 1; j < 4; j++){
        double minN = 0;
        double maxN = 1;//normalized min and max limits
        double minD = 10000000;
        double maxD = 0;//min and max of data set
        for(int i = 0; i < 20; i++){
            if(data[i][j] < minD)
                minD = data[i][j];
            if(data[i][j] > maxD)
                maxD = data[i][j];//find min and max of dataset
        }
        for(int i = 0; i < 20; i++){//actual normalization of data
            data[i][j] = (data[i][j] - minD)/(maxD - minD);
        }
    }
}

void gradient(double theta[5], double alpha){//updated
    double phi[] = {0, 0, 0, 0, 0};
    double N = 15;
    for(int i = 0; i < N; i++){
        phi[0] = phi[0] + (-2/N) *(data[i][0] - (theta[0] + theta[1]*data[i][1] + theta[2]*data[i][2]
                                                 + theta[3]*data[i][3]
                                                 + theta[4]*data[i][4]));
        phi[1] = phi[1] + (-2/N) * data[i][1] * (data[i][0] - (theta[0] + theta[1]*data[i][1] + theta[2]*data[i][2]
                                                 + theta[3]*data[i][3]
                                                 + theta[4]*data[i][4]));
        phi[2] = phi[2] + (-2/N) * data[i][2] * (data[i][0] - (theta[0] + theta[1]*data[i][1] + theta[2]*data[i][2]
                                                 + theta[3]*data[i][3]
                                                 + theta[4]*data[i][4]));
        phi[3] = phi[3] + (-2/N) * data[i][3] * (data[i][0] - (theta[0] + theta[1]*data[i][1] + theta[2]*data[i][2]
                                                 + theta[3]*data[i][3]
                                                 + theta[4]*data[i][4]));
        phi[4] = phi[4] + (-2/N) * data[i][4] * (data[i][0] - (theta[0] + theta[1]*data[i][1] + theta[2]*data[i][2]
                                                 + theta[3]*data[i][3]
                                                 + theta[4]*data[i][4]));
    }
    for(int j = 0; j < 5; j++){
        theta[j] = theta[j] - (alpha * phi[j]);
    }
}

void runner(double theta[5], double alpha, int iterations){
    for(int i = 0; i < iterations; i++){
        gradient(theta, alpha);
    }
}
