#include <para.h>

vector<string> list_folder(string path){
    vector<string> folders;
    DIR *dir = opendir(path.c_str());
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
         if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0))
         {
             string folder_path = path + "/" + string(entry->d_name);
             folders.push_back(folder_path);
         }
    }
    closedir(dir);
    return folders;

}

vector<string> list_file(string folder_path){
    vector<string> files;
    DIR *dir = opendir(folder_path.c_str());
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
         if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0))
         {
             string file_path = folder_path + "/" + string(entry->d_name);
             files.push_back(file_path);
         }
    }
    closedir(dir);
    return files;
}

void calculateNum(vector<string> folders, int &number_of_class, int &number_of_sample, int &number_of_feature){
    number_of_class=folders.size();
    Mat src, dst;
    for(size_t i=0; i<folders.size(); i++){

        vector<string> files = list_file(folders.at(i));
        number_of_sample += files.size();
        src = imread(files.at(0));
        resize(src, src, Size(36,36));
        cvtColor(src, dst, CV_BGR2GRAY);
        vector<float> feature;
        hog.compute(dst, feature);
        number_of_feature=feature.size();
    }
}

int main(int argc, char *argv[]){

    Mat src, dst;
    int number_of_class, number_of_sample, number_of_feature;
    int index = 0;

    Ptr<ml::SVM> svm = ml::SVM::create();
    //svm->setGamma(0.50625);
    //svm->setC(2.67);
    svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);

    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    vector<string> folders = list_folder(PATH1);
    calculateNum(folders, number_of_class, number_of_sample, number_of_feature);
    cout<<number_of_class<<" "<<number_of_sample<<" "<<number_of_feature<<endl;

    Mat data = Mat(number_of_sample, number_of_feature, CV_32FC1);
    Mat label = Mat(number_of_sample, 1, CV_32SC1);

    for(size_t i=0; i<folders.size(); i++){

        cout<<i<<"..."<<endl;
        vector<string> files = list_file(folders.at(i));
        for(size_t j=0; j<files.size(); j++){
            src = imread(files.at(j));
            resize(src, src, Size(36,36));
            cvtColor(src, dst, CV_BGR2GRAY);
            if(src.empty())
                continue;
            vector<float> feature;
            hog.compute(dst, feature);
            if(feature.size() < number_of_feature){
                cout<<"error"<<endl;
                continue;
            }

            for(size_t t=0; t<number_of_feature; t++)
                data.at<float>(index, t) = feature.at(t);

            label.at<int>(index, 0) = i;
            //cout<<i<<endl;
            index++;
        }
    }
    Ptr<TrainData> td = TrainData::create(data, ROW_SAMPLE, label);
    svm->train(td);
    svm->save(SAVE1);

    /*Bien xanh
     *0-> Bien 02
     *1-> Bien 08
     *2-> Bien 03
     *3-> Negative
     *4-> Bien 09  */


    /*Bien do
     *0->01
     *1->05
     *2->04
     *3->Negative
     *4->07
     *5->06
    */

    /*Ptr<SVM> svm = Algorithm::load<SVM>(LOAD1);

    Mat dst1;
    Mat src1 = imread(TEST1);
    Mat data1 = Mat(1 , 324, CV_32FC1);
    vector<float> feature;
    resize(src1, src1, Size(36,36));
    cvtColor(src1, dst1, CV_BGR2GRAY);
    hog.compute(dst1, feature);
    for(size_t t=0; t<324; t++)
        data1.at<float>(0, t) = feature.at(t);
    cout<<svm->predict(data1);*/
    waitKey(0);
    return 0;
}
