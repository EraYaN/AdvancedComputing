#include "infoli.h"

//int ReadFileLine(char *iAppBuf, int iAppBufSize, FILE *pInFile, user_float_t *iAppArray){
//    //FIXME: make this function more robust
//    char *strNumber;
//    int i = 0;
//    //Get one line
//    if(fgets(iAppBuf, iAppBufSize, pInFile)){
//        //Convert the ASCII string of one element to a double precision floating point value
//		rsize_t strmax = sizeof iAppBuf;
//		char *next_token;
//        strNumber = strtok_s(iAppBuf, &strmax, " ", &next_token);
//        i = 0;
//        //printf("Line:\n");
//        while ((strNumber != NULL) && (i<IO_NETWORK_SIZE)){
//            iAppArray[i] = atof(strNumber);//atof() should change if using integers or fixed point
//            //printf ("(%s) %0.2f ", strNumber, iAppArray[i]);
//            strNumber = strtok_s(NULL, " ");
//            i++;
//        }
//        //printf("i: %d\n", i);
//        if(i<IO_NETWORK_SIZE){
//            //BUG: if only one element is missing but the line ends in a space, the error is not detected
//            printf("Error: Input line doesn't have enough elements, only %d\n", i);
//            exit(EXIT_FAILURE);
//        }
//        return 1;//success
//    }else{
//        if(!feof(pInFile)){
//        printf("Error: Reading from input file didn't finish successfully\n");
//        exit(EXIT_FAILURE);
//        }
//        return 0;//end of file
//    }
//}