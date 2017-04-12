#include"som.h"

/*
 TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
xxxx     unsigned byte   ??               label

The labels values are 0 to 9. 


 TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel


 TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label


The labels values are 0 to 9.
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
*/


enum MNIST_format{
NB_VECTOR,
NB_ROW,
NB_COLUMN
};

int * genSuffledVector(int nbl){ // generer un vecteur avec des valeur de 0 à nbl
	int rd,tmp,i;
	int * vec;
	vec=(int *)malloc(sizeof(int)*nbl);
	
	for(i=0;i<nbl;i++)
		vec[i]=i;
	
	for(i=0;i<nbl;i++){
		rd=rand()%nbl;
		tmp=vec[rd];
		vec[rd]=vec[i];
		vec[i]=tmp;
	}
	return vec;


}


void normalisationVector(double * vec,int nbl,double norm){ // rammener les valeurs de vecteur entre 0.0 - 1.0
	int i;
	for(i=0;i<nbl;i++)
		vec[i]/=norm;	


}

double norm(double * vec,int nbl){ // calculer la norm d'un vecteur
	int i;
	double res=0.0;
	for(i=0;i<nbl;i++)
		res+=vec[i]*vec[i];	

	return res/(double)nbl;

}

double randomNumber(double ecart){ // generer un nombre real entre 0.0 - ecart
	return ecart*((double)rand() / (double)RAND_MAX);
}


double * averageVector(data_v * data, int nbLigne, int data_len) {
   	int i,j;
	double * averages;
	averages=(double *)malloc(sizeof(double)*data_len);
	memset(averages,0,sizeof(double)*data_len);
	for(j=0;j<nbLigne;j++)
		for(i=0;i<data_len;i++)
			averages[i]+=data[j].v[i];
	
for(i=0;i<data_len;i++)
		averages[i]=(double)averages[i]/(double)nbLigne;
	return averages;
} 



double distanceEuclid(double * p, double * q,int nbl){
	double result=0.0;
	int i;
		for(i=0;i<nbl;i++)
			result+=(p[i] - q[i])*(p[i] - q[i]);
		
	return (double)sqrt(result);

}




void initData(data_base * db,char * fileName){
	FILE *fp;
	int header;
	int c;
	int i,j,bit;
	int val=0;
	fp = fopen(fileName,"rb");
	if(fp == NULL){
		perror("Error in opening file");
		return;
	}
	val=0;
	val|=fgetc(fp)<<24;
	val|=fgetc(fp)<<16;
	val|=fgetc(fp)<<8;
	val|=fgetc(fp);
	switch(val){
		case 2051:
        	printf("Init data\n");
			for(i=0,val=0,bit=0,header=0,db->data_len=1; i < 12;i++){
			   c = fgetc(fp);
			   val|=(c<<(8*(3-bit++)));
				if(bit==4){
					if(header==NB_VECTOR){
						db->data_nbl=val;	
						printf("nombre d'image : %d\n",val);
					}else if(header==NB_ROW){
						db->data_len*=val;
						db->data_w=val;
						printf("pixel width : %d\n",val);
					}else if(header==NB_COLUMN){
						db->data_len*=val;
						db->data_h=val;
						printf("pixel height : %d\n",val);	
					}
					header++;	
					bit=0;	
					val=0;
				}
			}

			db->data=(data_v *)malloc(sizeof(data_v)*db->data_nbl);

			for(j=0;1;j++){
				db->data[j].v=(double *)malloc(sizeof(double)*db->data_len);
				for(i=0;i<db->data_len;i++){
					c=fgetc(fp);
//                    if('\n'==c)            printf("nnnn\n");	

					db->data[j].v[i]=c/(double)255;
				}
				db->data[j].norm=norm(db->data[j].v,db->data_len); 
			//	normalisation(db->data[j].v,db->data_len,db->data[j].norm); 
				if(c==EOF)break;		
			}  
            
			db->suffled_index=genSuffledVector(db->data_nbl);
		break;

		case 2049:
            val=0;
            val|=fgetc(fp)<<24;
//            printf("%d\n",val);	
	        val|=fgetc(fp)<<16;
//            printf("%d\n",val);	
	        val|=fgetc(fp)<<8;
//            printf("%d\n",val);	
       	        val|=fgetc(fp);
            printf("NB Labels %d\n",val);	

    		for(j=0;j<val;j++){
                db->data[j].label=fgetc(fp);
           // printf("%d\n",j);	
			} 	
    	break;
	
		default:
		break;
	}
	fclose(fp);
}


void initNetwork(data_base * db,network * net,parametre * pm){
	
	int i,j;	
	double * average;
	net->nodes=(node *)malloc(sizeof(node)*net->nb_nodes);
	average=averageVector(db->data,db->data_nbl, db->data_len);
	printf("\nInit network\n");
	
	printf("reseaux width : %d\n",net->width);
	printf("reseaux height : %d\n",net->height);
	printf("Random ecart : %.2f%c\n",pm->random_ecart*100,'%');
	
	for (i = 0; i < net->nb_nodes; i++){
		net->nodes[i].weight=(double *)malloc(sizeof(double)*db->data_len);
		
		for (j = 0; j < db->data_len; j++){
			net->nodes[i].weight[j]=average[j]-pm->random_ecart+randomNumber(pm->random_ecart*2);
			if(net->nodes[i].weight[j] < 0.0)
				net->nodes[i].weight[j]=0.0;
			if(net->nodes[i].weight[j] > 1.0)
				net->nodes[i].weight[j]=1.0;
		}

		net->nodes[i].activation=0.0;
		net->nodes[i].label = -1;
	}
	free(average);
	
}


void initBMU(network * net, best_matching_unit_Header *bmu){
	best_matching_unit * nouveau;
	int mini=0;
	int i;
	for(i=1;i<net->nb_nodes;i++){
		if(net->nodes[mini].activation>net->nodes[i].activation)
			mini=i;
	}
	
	
	bmu->nbl=0;
	bmu->begin=NULL;
	
	
	for(i=0;i<net->nb_nodes;i++){
		if(net->nodes[mini].activation==net->nodes[i].activation){
			nouveau=(best_matching_unit *)malloc(sizeof(best_matching_unit));
			nouveau->minX=i%net->width;
			nouveau->minY=i/net->width;
			nouveau->next=bmu->begin;
			bmu->begin=nouveau;
			if(bmu->nbl==0)
				bmu->end=bmu->begin;
			bmu->nbl++;

		}
	}
	
	
}
void freeBMU(best_matching_unit_Header *bmu){
	best_matching_unit * tmp;
	tmp=bmu->begin;
	while(tmp!=bmu->end){
		if(tmp)
			free(tmp);
		tmp=tmp->next;
	}
	bmu->nbl=0;
}
void freeData(data_base * db,data_base * db2,network * net){
int i;
for(i=0;i<db->data_len;i++)
	free(db->data[i].v);
free(db->data);

for(i=0;i<db2->data_len;i++)
	free(db2->data[i].v);
free(db2->data);

for (i = 0; i < net->nb_nodes; i++)
	free(net->nodes[i].weight);	
free(net->nodes);
}

void parametrage(parametre * pm){
	pm->alpha=(double)pm->alpha_init*(1.00-(pm->it_current/(double)pm->it_total));
	pm->rayon=(int)ceil(pm->rayon_init*(1.00-(pm->it_current/(double)pm->it_total)));
}

void voisinage(network * net,parametre * pm, best_matching_unit_Header * bmu, data_v * randomData,int nb_data_v){
	int i,j,k,x,y,r=pm->rayon;
	best_matching_unit * winner;
	int randomNumber=rand()%bmu->nbl;
	winner=bmu->begin;
	for(i=0;i<bmu->nbl;i++){
		if(i==randomNumber){
			break;
		}				
		winner=winner->next;
	}

	
	for(i=-r;i<=r;i++)
		for(j=-r;j<=r;j++)
			for(k=0;k<nb_data_v;k++){
				x=winner->minX+j;
				y=winner->minY+i;	
				if(x<net->width&&y<net->height&&x>=0&&y>=0)
					net->nodes[x+y*net->width].weight[k]+=pm->alpha*(randomData->v[k]-net->nodes[x+y*net->width].weight[k]);
			}
//	printf("%d %d\n",winner->minX,winner->minY);

}

void sauvegardeImage(char *filename, data_base * db, network * net){
    FILE *fp;
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%lu %lu\n",(long unsigned int)net->width*db->data_w,(long unsigned int)db->data_h*net->height);
    fprintf(fp, "%d\n",255);
	int i,j,k,l;
    // appretisage peut faire atteindre 0 < pixel , 1 > pixel , on fix min 0.0 et max 1.0 
 	for (i = 0; i < net->nb_nodes; i++){
		for (j = 0; j < db->data_len; j++){
			if(net->nodes[i].weight[j] < 0.0)
				net->nodes[i].weight[j]=0.0;
			if(net->nodes[i].weight[j] > 1.0)
				net->nodes[i].weight[j]=1.0;
		}
	}

	for(l=0;l<net->height;l++)
		for(k=0;k<db->data_h;k++)
				for(j=0;j<net->width;j++)
				for(i=0;i<db->data_w;i++){
				fputc((char)(net->nodes[j+l*net->width].weight[i+k*db->data_w]*255), fp);
	    		fputc((char)(net->nodes[j+l*net->width].weight[i+k*db->data_w]*255), fp);
	    		fputc((char)(net->nodes[j+l*net->width].weight[i+k*db->data_w]*255), fp);
	    
		}
    fclose(fp);
}

void apprentisage(data_base * db,network * net,parametre * pm){
	int i,j;
	int range;
	char buf[100];	
	double reste,reste2;
	data_v * randomData;
	best_matching_unit_Header bmu;
	printf("\nApprentisage\n");
	range=(int)(pm->training_range*db->data_nbl);

  	clock_t t,pass;
	pass=1;
    for(;pm->it_current < pm->it_total;pm->it_current++){
		t = clock();
 		parametrage(pm);	
		printf("progress: %.2f%c(%d/%d) parcour: %d rayon: %d alpha: %f\n",(pm->it_current/(double)(pm->it_total))*100,'%',pm->it_current,pm->it_total,range,pm->rayon,pm->alpha);
		for(j=0;j < range;j++){
			randomData=&db->data[db->suffled_index[rand()%db->data_nbl]];
			for(i=0;i<net->nb_nodes;i++)
				net->nodes[i].activation=distanceEuclid(net->nodes[i].weight,randomData->v, db->data_len);
			initBMU(net,&bmu);
			voisinage(net,pm,&bmu,randomData,db->data_len);
			parametrage(pm);
			freeBMU(&bmu);
		}
 		t = clock() - t;
		pass+=t;
		reste2=(double)(pm->it_total-pm->it_current-1)*t/CLOCKS_PER_SEC;
		reste=(double)(pm->it_total-pm->it_current-1)*(pass/(float)(pm->it_current+1))/CLOCKS_PER_SEC;
		if(reste2>reste)
		 	printf ("iteration : %.2fs\nreste: %dh %dm %ds - %dh %dm %ds\n",(float)t/CLOCKS_PER_SEC,(int)(reste/60./60.),(int)(reste/60.)%60,(int)(reste)%60,(int)(reste2/60./60.),(int)(reste2/60.)%60,(int)(reste2)%60);
		else
	 	printf ("iteration : %.2fs\nreste: %dh %dm %ds - %dh %dm %ds\n",(float)t/CLOCKS_PER_SEC,(int)(reste2/60./60.),(int)(reste2/60.)%60,(int)(reste2)%60,(int)(reste/60./60.),(int)(reste/60.)%60,(int)(reste)%60);

 		sprintf (buf,"progress/it%d.ppm", pm->it_current);
	    sauvegardeImage(buf,db,net);

	}
	printf("sucess: %.2f%c(%d/%d) \n",(pm->it_current/(double)(pm->it_total))*100,'%',pm->it_current,pm->it_total);

}

void sauvegardeParametre(char *filename,parametre * pm){
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "Parametre\n");
	fprintf(fp, "net_w %d\n",pm->net_w);
	fprintf(fp, "net_h %d\n",pm->net_h);
    fprintf(fp, "iteration_total %d\n",pm->it_total);
	fprintf(fp, "rayon_init %d\n",pm->rayon_init);
	fprintf(fp, "alpha_init %lf\n",pm->alpha_init);
	fprintf(fp, "training_range %lf\n",pm->training_range);
	fprintf(fp, "random_ecart %lf\n",pm->random_ecart);
	fclose(fp);
}


void initParametreNetwork(network * net,parametre * pm){
	net->height=10;
	net->width=15;
	net->nb_nodes=net->width*net->height;
	pm->random_ecart=0.01;
}

void initParametreOrdonnancement(network * net,parametre * pm){
	int total;	
	pm->net_w=net->width;
	pm->net_h=net->height;
	pm->alpha=(double)0.2*((double)rand() / (double)RAND_MAX)+0.7;
	pm->alpha_init=pm->alpha;
	pm->training_range=0.1;
	pm->it_current=0;
	pm->it_total=5*(int)sqrt(pm->net_h*pm->net_w);
	for(pm->rayon=1,total=8;total<pm->net_h*pm->net_w/2;total+=pm->rayon*8,pm->rayon++);// 50% de node
	pm->rayon_init=pm->rayon;
	printf("\nPhase ordonnancement\n");
	printf("it total : %d\n",pm->it_total);
	printf("rayon init : %d\n",pm->rayon_init);
	printf("alpha init : %lf\n",pm->alpha_init);
	printf("parcours : %.2f%c de donnée totale\n",pm->training_range*100,'%');	
}

void initParametreRaffinage(network * net,parametre * pm){
	pm->net_w=net->width;
	pm->net_h=net->height;
	pm->rayon_init=pm->rayon=2;
	pm->alpha=0.1*((double)0.2*((double)rand() / (double)RAND_MAX)+0.7);
	pm->alpha_init=pm->alpha;
	pm->training_range=1.00;
	pm->it_current=0;
	pm->it_total=500*(int)sqrt(net->height*net->width);
	printf("\nPhase raffinage\n");
	printf("it total : %d\n",pm->it_total);
	printf("rayon init : %d\n",pm->rayon_init);
	printf("alpha init : %lf\n",pm->alpha_init);
	printf("Parcours : %.2f%c de donnée totale\n",pm->training_range*100,'%');	

}




void printCarte(network * net){
int i,j;
	for(j=0;j<net->height;j++){
		for(i=0;i<net->width;i++)
		    if(net->nodes[i+j*net->width].label==-1)
		        printf(" ");
		    else		
		        printf("%d",net->nodes[i+j*net->width].label);
		printf("\n");
	}
}


void initEtiquet(data_base * db,network * net){
	int i,j;
	int max,min;
	float act,save;
	for(i=0;i<net->nb_nodes;i++){
		for(j=0,min=0,save=1000;j<db->data_nbl;j++){
			act=distanceEuclid(db->data[j].v, net->nodes[i].weight, db->data_len);
			if(save == act)
				printf("%d %d \n",db->data[min].label,db->data[j].label);
	
			if(save > act){
				save=act;
				min=j;
			}	
		}
		net->nodes[i].label=db->data[min].label;
	}
}


void verifierErreur(data_base * db_verify,network * net){
	int i,j,k;
    int vrai=0,faux=0;
	int max,min;
	float act,save;
		for(j=0;j<db_verify->data_nbl;j++){
        	for(i=0,min=0,save=1000;i<net->nb_nodes;i++){
			act=distanceEuclid(db_verify->data[j].v, net->nodes[i].weight, db_verify->data_len);
			if(save == act)
				printf("BMU : %d %d \n",db_verify->data[min].label,db_verify->data[j].label);
	
			if(save > act){
				save=act;
				min=j;
			}	
		}
        if(net->nodes[min].label==db_verify->data[j].label)
		    vrai++;
        else
		    faux++;
	}
    printf("les erreurs : %lf\n",faux/(double)db_verify->data_nbl);
}




int main (){
// DECLARATION VARIABLE
  data_base db;
  data_base db_verify;
  network net;
  parametre phase1;
  parametre phase2;
// DECLARATION VARIABLE

// INIT DONNEE
  initData(&db,"train-images-idx3-ubyte");
  initData(&db,"train-labels-idx1-ubyte");
  initData(&db_verify,"t10k-images-idx3-ubyte");
  initData(&db_verify,"t10k-labels-idx1-ubyte");
  initParametreNetwork(&net,&phase1);
  initNetwork(&db,&net,&phase1);
// INIT DONNEE


// APPRENTISAGE ORDONNANCEMENT
  initParametreOrdonnancement(&net,&phase1);  
  apprentisage(&db,&net,&phase1);
  sauvegardeImage("phase1.ppm",&db,&net);
  initEtiquet(&db,&net); 
  printCarte(&net);
// APPRENTISAGE ORDONNANCEMENT


// APPRENTISAGE RAFFIANGE
  initParametreRaffinage(&net,&phase2);
  apprentisage(&db,&net,&phase2);
  sauvegardeImage("phase2.ppm",&db,&net);
// APPRENTISAGE RAFFIANGE
  

// DONNER LES ETIQUETS
  initEtiquet(&db,&net);
// DONNER LES ETIQUETS


// VERIFIER LES ERREURS
  verifierErreur(&db_verify,&net);
// VERIFIER LES ERREURS


// SAUVGARDE DE PARAMETRE & LIBRATION DE MEMOIRE 
  sauvegardeParametre("phase1.txt",&phase1);
  sauvegardeParametre("phase2.txt",&phase2);
  freeData(&db,&db_verify,&net);
// SAUVGARDE DE PARAMETRE & LIBRATION DE MEMOIRE

  return 0;
}

