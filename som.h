#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

struct data_v{
  double * v;
  int label;
  double norm;
};
typedef struct data_v data_v;

struct data_base {
	int data_h;
	int data_w;
	int data_len;
	int data_nbl;
	data_v * data;
	int * suffled_index;

	int nb_dictionnary;
	char ** dictionnary;
};
typedef struct data_base data_base;


struct node {
	double * weight; // 4;
	double activation; // distance uclidien
	int label;
};
typedef struct node node;

struct parametre{
	int it_current;
	int it_total;
	int rayon;
	int rayon_init;
	int net_w;
	int net_h;

	double training_range;
	double alpha;
	double alpha_init;
	double random_ecart;
};
typedef struct parametre parametre;

struct network {
	int width;
	int height;
	int nb_nodes;
	node * nodes;// node ** 50% DE
};
typedef struct network network;

struct  best_matching_unit {
	int minX;
	int minY;
	struct best_matching_unit * next;
};
typedef struct best_matching_unit best_matching_unit;


struct best_matching_unit_Header{
	int nbl;
	struct best_matching_unit * begin;
	struct best_matching_unit * end;
};
typedef struct best_matching_unit_Header best_matching_unit_Header;



