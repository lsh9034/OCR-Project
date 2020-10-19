#define _CRT_SECURE_NO_WARNINGS
#define unicd_h 0xAC00
#define BLACK 0
#define WHITE -1
#define TRUE 1
#define p2(x,y) Point_2(x,y)
#define ar(x,y) Area(x,y)
#define binary(in , out, x, y) adaptiveThreshold(in, out, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, x, y)
#include <vector>
#include <list>
#include <algorithm>
#include <opencv.hpp>
#include "Hangul.h"
#include <locale.h>
#include<stdio.h>
using namespace std;
using namespace cv;
using namespace OCR;

struct Point_2{
	int y;
	int x;
	Point_2(int _y = 0, int _x = 0) :y(_y), x(_x){};
};

struct Area{
	int		bold;
	bool	check;
	Point_2	top_left;
	Point_2	bottom_right;
	Area(Point_2 lt, Point_2 rb) :top_left(lt), bottom_right(rb){ bold = 0,check = false; }
	void	print()				{ printf("top_left, bottom_right : (%d, %d), (%d, %d), bold : %d\n", top(), left(), bottom(), right(), bold); }
	int&	left()			{ return top_left.x; }
	int&	right()			{ return bottom_right.x; }
	int&	top()			{ return top_left.y; }
	int&	bottom()		{ return bottom_right.y; }
	int		b_left()		{ return left() - bold; }
	int		b_right()		{ return right() + bold; }
	int		b_top()			{ return top() - bold; }
	int		b_bottom()		{ return bottom() + bold; }
	int		size()			{ return b_right() - b_left() + 1; }
	void	left(int _left)		{ left() = _left; }
	void	right(int _right)	{ right() = _right; }
	void	top(int _top)		{ top() = _top; }
	void	bottom(int _bottom)	{ bottom() = _bottom; }
};

//영역의 이미지의 두께를 구한다
void Area_Bold(Mat &base, vector<Area> &area){
	for (int i = 0; i < area.size(); i++){
		int bottom	= area[i].bottom() + 1;
		int right	= area[i].right() + 1;
		int w_bold	= -1;
		int h_bold	= -1;
		for (int r = area[i].top(); r < bottom; r++){
			int bold = 0;
			for (int c = area[i].left(); c < right; c++){
				if (base.at<char>(r, c) == BLACK) bold++;
				else if (bold > 0){
					if (bold < w_bold || w_bold == -1) w_bold = bold;
					bold = 0;
				}
			}
			if ((bold < w_bold || w_bold == -1) && bold > 0) w_bold = bold;
		}
		for (int c = area[i].left(); c < right; c++){
			int bold = 0;
			for (int r = area[i].top(); r < bottom; r++){
				if (base.at<char>(r, c) == BLACK) bold++;
				else if (bold > 0){
					if (bold < h_bold || h_bold == -1) h_bold = bold;
					bold = 0;
				}
			}
			if ((bold < h_bold || h_bold == -1) && bold > 0) h_bold = bold;
		}
		if (w_bold < 0) w_bold = 0;
		if (h_bold < 0) h_bold = 0;
		if (w_bold < h_bold && w_bold > 0) area[i].bold = w_bold;
		else area[i].bold = h_bold;
	}
}

void BFS(Mat &base, Mat &change, vector<Area> &store){
	const	int		direction		= 8;
	int				cnt				= 0;
	int				C[direction]	= { 0, 1, 1, 1, 0, -1, -1, -1 };
	int				R[direction]	= { -1, -1, 0, 1, 1, 1, 0, -1 };
	Mat				check;
	list<Point_2>	bfs;
	check.create(base.size(), CV_8UC1);
	store.clear();
	for (int r = 0; r < base.rows; r++){
		for (int c = 0; c < base.cols; c++){
			if (base.at<char>(r, c) == WHITE && change.at<char>(r, c) == WHITE) continue;
			if (check.at<char>(r, c) == TRUE) continue;
			bfs.push_back(Point_2(r, c));
			check.at<char>(r, c) = TRUE;
			Area temp(p2(base.rows, base.cols), p2(-1, -1));
			while (1){
				if (bfs.empty())
					break;
				Point_2 p = bfs.front();
				bfs.pop_front();
				int r = p.y;
				int c = p.x;
				if (base.at<char>(r, c) == BLACK){
					if (r < temp.top()) temp.top(r);		//현재의 r의 값이 top보다 작으면 저장
					if (r > temp.bottom()) temp.bottom(r);	//현재의 r의 값이 bottom보다 크면 저장
					if (c < temp.left()) temp.left(c);		//현재의 c의 값이 left보다 작으면 저장
					if (c > temp.right()) temp.right(c);	//현재의 c의 값이 right보다크면 저장
				}
				for (int d = 0; d < direction; d++){
					int _r = r + R[d];
					int _c = c + C[d];
					if (_r < 0 || _r >= base.rows) continue;
					if (_c < 0 || _c >= base.cols) continue;
					if (check.at<char>(_r, _c) == TRUE) continue;
					if (base.at<char>(_r, _c) == BLACK || change.at<char>(_r, _c) == BLACK){
						check.at<char>(_r, _c) = TRUE;
						bfs.push_back(Point_2(_r, _c));
					}
				}
			}
			if (temp.left()>temp.right() || temp.top()>temp.bottom()) continue;
			store.push_back(temp);
		}
	}
	Area_Bold(base, store);
}

//원본 이미지에서 영역헤 해당되는 부분의 이미지를 사본 이미지로 바꾼다
void Change_Image(Mat &change, Mat &base, Area area){
	int	top = area.top();
	int	left = area.left();
	int	rows = area.bottom() - top + 1;
	int	cols = area.right() - left + 1;
	int bold = area.bold;
	int r_size = base.rows;
	int c_size = base.cols;
	for (int r = 0; r < rows + bold * 2; r++){
		for (int c = 0; c < cols + bold * 2; c++){
			int _r = r + top - bold;
			int _c = c + left - bold;
			if (_r < 0 || _c < 0) continue;
			if (_r >= r_size|| _c >= c_size) continue;
			if (base.at<char>(_r, _c) > change.at<char>(r, c) || base.at<char>(_r, _c) == -1)
				base.at<char>(_r, _c) = change.at<char>(r, c);
		}
	}
}

void Area_Clear(Mat &input){
	int rows = input.rows;
	int cols = input.cols;
	for (int r = 0; r < rows; r++)
		for (int c = 0; c < cols; c++)
			input.at<char>(r, c) = WHITE;
}

//영역의 이미지를 따온다
void Area_Image(Mat &input, Mat &save, Area area){
	int	top = area.top();
	int	left = area.left();
	int	rows = area.bottom() - top + 1;
	int	cols = area.right() - left + 1;
	int	type = input.type();
	int	bold = area.bold;
	save.create(rows + bold * 2, cols + bold * 2, type);
	Area_Clear(save);
	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++)
			save.at<char>(r + bold, c + bold) = input.at<char>(r + top, c + left);
	}
}

//영역 모폴로지 함수
void Area_Morphology(Mat &base, Mat &change, vector<Area> &area){
	for (int i = 1; i < area.size(); i++){
		Mat temp;
		int k = area[i].bold * 2;
		if (k % 2 == 0) k = k + 1;
		Mat element(k, k, CV_8U, Scalar(1));
		Area_Image(base, temp, area[i]);
		morphologyEx(temp, temp, MORPH_OPEN, element);
		Change_Image(temp, change, area[i]);
	}
}

//영역에 겹치는 부분이 있는지 확인한다
bool Is_Area_Covered(Area &i, Area &j){
	if ((i.right() >= j.left() && i.left() <= j.right()) && (i.bottom() >= j.top() && i.top() <= j.bottom())) return true;
	return false;
}

//합친 영역의 비율이 정사각형에 더 가까운지 비교한다
bool Is_Better_Combine(Area &i, Area &j, double &rate){
	int		width	= i.right() - i.left() + 1;
	int		height	= i.bottom() - i.top() + 1;
	double	rate1	= (width > height) ? (double)((double)width / (double)height) : (double)((double)height / (double)width);
	int		left	= (i.left() < j.left()) ? i.left() : j.left();
	int		right	= (i.right() > j.right()) ? i.right() : j.right();
	int		top		= (i.top() < j.top()) ? i.top() : j.top();
	int		bottom	= (i.bottom() > j.bottom()) ? i.bottom() : j.bottom();
	width			= right - left + 1;
	height			= bottom - top + 1;
	double	rate2	= (width > height) ? (double)((double)width / (double)height) : (double)((double)height / (double)width);
	rate = rate2;
	return	rate1 > rate2;//합친 영역의 비율이 합치기 전 영역의 비율보다 작으면 참
}

//영역을 교체하는 함수
void Area_Change(Area &area, Area &compare){
	if (area.top() > compare.top())
		area.top(compare.top());
	if (area.left() > compare.left())
		area.left(compare.left());
	if (area.bottom() < compare.bottom())
		area.bottom(compare.bottom());
	if (area.right() < compare.right())
		area.right(compare.right());
	if (area.bold < compare.bold)
		area.bold = compare.bold;
}

bool Area_Compare(Area &i, Area &j){
	if (i.top() > j.bottom()) return false;
	if (i.bottom() < j.top()) return true;
	return i.left()<j.left();
}

//Is_Area_Covered 함수를 통하여 영역을 합친다
void Combine_Word_Covered(vector<Area> &input, vector<Area> &output){
	vector<Area> temp;
	for (int i = 0; i < input.size(); i++){
		if (input[i].check == true) continue;
		temp.push_back(input[i]);
		for (int j = i + 1; j < input.size(); j++){
			if (input[j].check == true) continue;
			if (Is_Area_Covered(temp.back(), input[j])){
				Area_Change(temp.back(), input[j]);
				input[j].check = true;
			}
		}
	}
	for (int i = 0; i < temp.size(); i++)
		temp[i].check = false;
	output = temp;
}

//Is_Better_Combine 함수를 통하여 영역을 합친다
void Combine_Word_Better(vector<Area> &input, vector<Area> &output){
	vector<Area> temp;
	for (int i = 0; i < input.size(); i++){
		if (input[i].check == true) continue;
		temp.push_back(input[i]);
		double	rate = INT_MAX;
		int		check = -1;
		for (int j = 0; j < input.size(); j++){
			if (input[j].check == true || j == i) continue;
			Area t = temp.back();
			Area_Change(t, input[j]);
			int count = 0;
			for (int k = 0; k < input.size(); k++){
				if (k == i || k == j) continue;
				if (Is_Area_Covered(t, input[k]))
					count++;
			}
			if (count > 1) continue;
			double t_rate;
			if (Is_Better_Combine(temp.back(), input[j], t_rate)){
				if (t_rate > rate) continue;
				rate = t_rate;
				check = j;
			}
		}
		if (check == -1)continue;
		Area_Change(temp.back(), input[check]);
		input[check].check = true;
	}
	for (int i = 0; i < temp.size(); i++)
		temp[i].check = false;
	output = temp;
}

//Combine 함수들의 통하여 글자를 만든다
void Make_Word(vector<Area> &input, vector<Area> &output){
	int bfr_cnt = INT_MAX;
	sort(input.begin(), input.end(), Area_Compare);
	while (bfr_cnt > input.size()){
		bfr_cnt = input.size();
		Combine_Word_Covered(input, input);
	}
	sort(input.begin(), input.end(), Area_Compare);
	Combine_Word_Better(input, input);
	while (bfr_cnt > input.size()){
		bfr_cnt = input.size();
		Combine_Word_Covered(input, input);
	}
	sort(input.begin(), input.end(), Area_Compare);
}

void Image_To_Num(Mat& image, double* save){
	for (int r = 0; r < image.rows; r++){
		for (int c = 0; c < image.cols; c++){
			save[r * 16 + c] = (double)image.at<char>(r, c);
		}
	}
}

void Black_White(Mat& base){
	for (int r = 0; r < base.rows; r++)
		for (int c = 0; c < base.cols; c++){
			if (base.at<char>(r, c) > WHITE) base.at<char>(r, c) = BLACK;
			else if (base.at<char>(r, c) <= -100) base.at<char>(r, c) = BLACK;
			else base.at<char>(r, c) = WHITE;
		}
}

int main()
{
	char input[100] = { 0, };
	scanf("%s", &input);
	Hangul_s han_s(true);
	_wsetlocale(LC_ALL, L"korean");    //wprintf()를 사용하여 유니코드 중 한글을 사용하겠음을 의미.
	vector<Area> area;
	String image_name = input;//12,14,21,17
	Mat base = imread(image_name, IMREAD_GRAYSCALE);
	Mat change = imread(image_name, IMREAD_GRAYSCALE);
	Black_White(base);
	Black_White(change);
	/*FILE* input = fopen("Hangul.txt", "r");
	FILE* output = fopen("NewHangul.txt", "w");
	int type = base.type();
	char answer[10] = { 0, };
	for (int k = 0; k < 11172; k++)
	{
		Mat temp, temp2;
		double pixel[256] = { 0, };
		temp.create(Size(16, 16), type);
		temp2.create(Size(16, 16), type);
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				int p;
				fscanf(input, "%d", &p);
				temp.at<char>(i, j) = p;
			}
		}
		fscanf(input, "%s", &answer);
		BFS(temp, temp, area);
		for (int j = 0; j < area.size(); j++)
			Area_Change(area[0], area[j]);
		Area_Image(temp, temp2, area[0]);
		resize(temp2, temp2, Size(16, 16), 0, 0, 0);
		Black_White(temp2);
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				fprintf(output, "%3d", (int)temp2.at<char>(i, j));
			}
			fprintf(output, "\n");
		}
		fprintf(output, "%s\n", answer);
	}*///Hangul에서 만든데이터 사진으로 만들어서 다시 인식해보면서 나온 오차를 수정하기 위해 새로 데이터만든소스x`
	BFS(base, base, area);
	Area_Morphology(base, change, area);
	area.clear();
	BFS(base, change, area);
	Make_Word(area, area);
	imshow("base", base);
	imshow("change", change);
	//han_s.Hangul_Learn();
	for (int i = 0; i < area.size(); i++){
		Mat temp;
		double* data = new double[256];
		Area_Image(base, temp, area[i]);//영역 이미지를 따낸다
		resize(temp, temp, Size(16, 16), 0, 0, 0);//이미지를 크기를 16*16으로 바꾼다
		Black_White(temp);//흑백 이미지로 변환
		Image_To_Num(temp, data);
		wchar_t word = (wchar_t)han_s.Word_Search(data);
		if (i > 0 && area[i].b_top() >= area[i - 1].b_bottom()) printf("\n");
		if (i > 0 && area[i].b_left() >= area[i - 1].b_right() +  area[i].size() / 3.5) printf(" ");
		wprintf(L"%wc", word);
	}
	printf("\n");
	waitKey();

	return 0;
}