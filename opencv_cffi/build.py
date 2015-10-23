import re

from cffi import FFI


ffi = FFI()


ffi.set_source(
    "_opencv",
    """
    #include "opencv/cv.h"
    #include "opencv/highgui.h"
    #include "opencv2/core/types_c.h"
    #include "opencv2/videoio/videoio_c.h"
    """,
    libraries=[
        "opencv_highgui",
        "opencv_objdetect",
        "opencv_videoio",
    ],
)


ffi.cdef(
    re.sub(
    r"\bCVAPI\(([^)]+)\)",
    r"\1", re.sub(
        r"\bCV_DEFAULT\([^)]+\)",
        "",
    """
typedef signed char schar;
typedef unsigned char uchar;

typedef struct CvMemBlock { ...; } CvMemBlock;
typedef struct CvMemStorage { ...; } CvMemStorage;
typedef struct CvMemStoragePos { ...; } CvMemStoragePos;

typedef struct CvSize
{
    int width;
    int height;
} CvSize;

/** @brief This is the "metatype" used *only* as a function parameter.

It denotes that the function accepts arrays of multiple types, such as IplImage*, CvMat* or even
CvSeq* sometimes. The particular array type is determined at runtime by analyzing the first 4
bytes of the header. In C++ interface the role of CvArr is played by InputArray and OutputArray.
 */
typedef void CvArr;

typedef struct _IplImage {
    int  nSize;             /**< sizeof(IplImage) */
    int  ID;                /**< version (=0)*/
    int  nChannels;         /**< Most of OpenCV functions support 1,2,3 or 4 channels */
    int  alphaChannel;      /**< Ignored by OpenCV */
    int  depth;             /**< Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                               IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported.  */
    char colorModel[4];     /**< Ignored by OpenCV */
    char channelSeq[4];     /**< ditto */
    int  dataOrder;         /**< 0 - interleaved color channels, 1 - separate color channels.
                               cvCreateImage can only create interleaved images */
    int  origin;            /**< 0 - top-left origin,
                               1 - bottom-left origin (Windows bitmaps style).  */
    int  align;             /**< Alignment of image rows (4 or 8).
                               OpenCV ignores it and uses widthStep instead.    */
    int  width;             /**< Image width in pixels.                           */
    int  height;            /**< Image height in pixels.                          */
    struct _IplROI *roi;    /**< Image ROI. If NULL, the whole image is selected. */
    struct _IplImage *maskROI;      /**< Must be NULL. */
    void  *imageId;                 /**< "           " */
    struct _IplTileInfo *tileInfo;  /**< "           " */
    int  imageSize;         /**< Image data size in bytes
                               (==image->height*image->widthStep
                               in case of interleaved data)*/
    char *imageData;        /**< Pointer to aligned image data.         */
    int  widthStep;         /**< Size of aligned image row in bytes.    */
    int  BorderMode[4];     /**< Ignored by OpenCV.                     */
    int  BorderConst[4];    /**< Ditto.                                 */
    char *imageDataOrigin;  /**< Pointer to very origin of image data
                               (not necessarily aligned) -
                               needed for correct deallocation */
} IplImage;
typedef struct _IplTileInfo IplTileInfo;
typedef struct _IplROI IplROI;

typedef struct CvSeqBlock
{
    struct CvSeqBlock*  prev; /**< Previous sequence block.                   */
    struct CvSeqBlock*  next; /**< Next sequence block.                       */
  int    start_index;         /**< Index of the first element in the block +  */
                              /**< sequence->first->start_index.              */
    int    count;             /**< Number of elements in the block.           */
    schar* data;              /**< Pointer to the first element of the block. */
}
CvSeqBlock;

typedef struct CvSeq {
    int       total;          /**< Total number of elements.            */  \
    int       elem_size;      /**< Size of sequence element in bytes.   */  \
    schar*    block_max;      /**< Maximal bound of the last block.     */  \
    schar*    ptr;            /**< Current write pointer.               */  \
    int       delta_elems;    /**< Grow seq this many at a time.        */  \
    CvMemStorage* storage;    /**< Where the seq is stored.             */  \
    CvSeqBlock* free_blocks;  /**< Free blocks list.                    */  \
    CvSeqBlock* first;        /**< Pointer to the first sequence block. */
    ...;
} CvSeq;

/*
@deprecated CvMat is now obsolete; consider using Mat instead.
 */
typedef struct CvMat { ...; } CvMat;
typedef struct CvMatND { ...; } CvMatND;
typedef struct CvSparseMat { ...; } CvSparseMat;
typedef struct CvSparseNode { ...; } CvSparseNode;
typedef struct CvSparseMatIterator { ...; } CvSparseMatIterator;

/** @sa Scalar_
 */
typedef struct CvScalar { double val[4]; } CvScalar;

typedef struct CvTermCriteria { ...; } CvTermCriteria;

/*************************************** CvRect *****************************************/
/** @sa Rect_ */
typedef struct CvRect
{
    int x;
    int y;
    int width;
    int height;

}
CvRect;

/******************************* CvPoint and variants ***********************************/

typedef struct CvPoint {
    int x;
    int y;
} CvPoint;

/****************************************************************************************\
*                   Arithmetic, logic and comparison operations                          *
\****************************************************************************************/

/** dst(mask) = src1(mask) + src2(mask) */
CVAPI(void)  cvAdd( const CvArr* src1, const CvArr* src2, CvArr* dst,
                    const CvArr* mask CV_DEFAULT(NULL));

/** dst(mask) = src(mask) + value */
CVAPI(void)  cvAddS( const CvArr* src, CvScalar value, CvArr* dst,
                     const CvArr* mask CV_DEFAULT(NULL));

/** dst(mask) = src1(mask) - src2(mask) */
CVAPI(void)  cvSub( const CvArr* src1, const CvArr* src2, CvArr* dst,
                    const CvArr* mask CV_DEFAULT(NULL));

/** dst(mask) = src(mask) - value = src(mask) + (-value) */
void  cvSubS( const CvArr* src, CvScalar value, CvArr* dst,
                         const CvArr* mask CV_DEFAULT(NULL));

/** dst(mask) = value - src(mask) */
CVAPI(void)  cvSubRS( const CvArr* src, CvScalar value, CvArr* dst,
                      const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src1(idx) * src2(idx) * scale
   (scaled element-wise multiplication of 2 arrays) */
CVAPI(void)  cvMul( const CvArr* src1, const CvArr* src2,
                    CvArr* dst, double scale CV_DEFAULT(1) );

/** element-wise division/inversion with scaling:
    dst(idx) = src1(idx) * scale / src2(idx)
    or dst(idx) = scale / src2(idx) if src1 == 0 */
CVAPI(void)  cvDiv( const CvArr* src1, const CvArr* src2,
                    CvArr* dst, double scale CV_DEFAULT(1));

/** dst = src1 * scale + src2 */
CVAPI(void)  cvScaleAdd( const CvArr* src1, CvScalar scale,
                         const CvArr* src2, CvArr* dst );

/** dst = src1 * alpha + src2 * beta + gamma */
CVAPI(void)  cvAddWeighted( const CvArr* src1, double alpha,
                            const CvArr* src2, double beta,
                            double gamma, CvArr* dst );

/** @brief Calculates the dot product of two arrays in Euclidean metrics.

The function calculates and returns the Euclidean dot product of two arrays.

\f[src1  \bullet src2 =  \sum _I ( \texttt{src1} (I)  \texttt{src2} (I))\f]

In the case of multiple channel arrays, the results for all channels are accumulated. In particular,
cvDotProduct(a,a) where a is a complex vector, will return \f$||\texttt{a}||^2\f$. The function can
process multi-dimensional arrays, row by row, layer by layer, and so on.
@param src1 The first source array
@param src2 The second source array
 */
CVAPI(double)  cvDotProduct( const CvArr* src1, const CvArr* src2 );

/** dst(idx) = src1(idx) & src2(idx) */
CVAPI(void) cvAnd( const CvArr* src1, const CvArr* src2,
                  CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src(idx) & value */
CVAPI(void) cvAndS( const CvArr* src, CvScalar value,
                   CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src1(idx) | src2(idx) */
CVAPI(void) cvOr( const CvArr* src1, const CvArr* src2,
                 CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src(idx) | value */
CVAPI(void) cvOrS( const CvArr* src, CvScalar value,
                  CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src1(idx) ^ src2(idx) */
CVAPI(void) cvXor( const CvArr* src1, const CvArr* src2,
                  CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = src(idx) ^ value */
CVAPI(void) cvXorS( const CvArr* src, CvScalar value,
                   CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

/** dst(idx) = ~src(idx) */
CVAPI(void) cvNot( const CvArr* src, CvArr* dst );

/** dst(idx) = lower(idx) <= src(idx) < upper(idx) */
CVAPI(void) cvInRange( const CvArr* src, const CvArr* lower,
                      const CvArr* upper, CvArr* dst );

/** dst(idx) = lower <= src(idx) < upper */
CVAPI(void) cvInRangeS( const CvArr* src, CvScalar lower,
                       CvScalar upper, CvArr* dst );

/** The comparison operation support single-channel arrays only.
   Destination image should be 8uC1 or 8sC1 */

/** dst(idx) = src1(idx) _cmp_op_ src2(idx) */
CVAPI(void) cvCmp( const CvArr* src1, const CvArr* src2, CvArr* dst, int cmp_op );

/** dst(idx) = src1(idx) _cmp_op_ value */
CVAPI(void) cvCmpS( const CvArr* src, double value, CvArr* dst, int cmp_op );

/** dst(idx) = min(src1(idx),src2(idx)) */
CVAPI(void) cvMin( const CvArr* src1, const CvArr* src2, CvArr* dst );

/** dst(idx) = max(src1(idx),src2(idx)) */
CVAPI(void) cvMax( const CvArr* src1, const CvArr* src2, CvArr* dst );

/** dst(idx) = min(src(idx),value) */
CVAPI(void) cvMinS( const CvArr* src, double value, CvArr* dst );

/** dst(idx) = max(src(idx),value) */
CVAPI(void) cvMaxS( const CvArr* src, double value, CvArr* dst );

/** dst(x,y,c) = abs(src1(x,y,c) - src2(x,y,c)) */
CVAPI(void) cvAbsDiff( const CvArr* src1, const CvArr* src2, CvArr* dst );

/** dst(x,y,c) = abs(src(x,y,c) - value(c)) */
CVAPI(void) cvAbsDiffS( const CvArr* src, CvArr* dst, CvScalar value );

/****************************************************************************************\
*          Array allocation, deallocation, initialization and access to elements         *
\****************************************************************************************/

/** `malloc` wrapper.
   If there is no enough memory, the function
   (as well as other OpenCV functions that call cvAlloc)
   raises an error. */
CVAPI(void*)  cvAlloc( size_t size );

/** `free` wrapper.
   Here and further all the memory releasing functions
   (that all call cvFree) take double pointer in order to
   to clear pointer to the data after releasing it.
   Passing pointer to NULL pointer is Ok: nothing happens in this case
*/
CVAPI(void)   cvFree_( void* ptr );

/** @brief Creates an image header but does not allocate the image data.

@param size Image width and height
@param depth Image depth (see cvCreateImage )
@param channels Number of channels (see cvCreateImage )
 */
CVAPI(IplImage*)  cvCreateImageHeader( CvSize size, int depth, int channels );

/** @brief Initializes an image header that was previously allocated.

The returned IplImage\* points to the initialized header.
@param image Image header to initialize
@param size Image width and height
@param depth Image depth (see cvCreateImage )
@param channels Number of channels (see cvCreateImage )
@param origin Top-left IPL_ORIGIN_TL or bottom-left IPL_ORIGIN_BL
@param align Alignment for image rows, typically 4 or 8 bytes
 */
CVAPI(IplImage*) cvInitImageHeader( IplImage* image, CvSize size, int depth,
                                   int channels, int origin CV_DEFAULT(0),
                                   int align CV_DEFAULT(4));

/** @brief Creates an image header and allocates the image data.

This function call is equivalent to the following code:
@code
    header = cvCreateImageHeader(size, depth, channels);
    cvCreateData(header);
@endcode
@param size Image width and height
@param depth Bit depth of image elements. See IplImage for valid depths.
@param channels Number of channels per pixel. See IplImage for details. This function only creates
images with interleaved channels.
 */
CVAPI(IplImage*)  cvCreateImage( CvSize size, int depth, int channels );

/** @brief Deallocates an image header.

This call is an analogue of :
@code
    if(image )
    {
        iplDeallocate(*image, IPL_IMAGE_HEADER | IPL_IMAGE_ROI);
        *image = 0;
    }
@endcode
but it does not use IPL functions by default (see the CV_TURN_ON_IPL_COMPATIBILITY macro).
@param image Double pointer to the image header
 */
CVAPI(void)  cvReleaseImageHeader( IplImage** image );

/** @brief Deallocates the image header and the image data.

This call is a shortened form of :
@code
    if(*image )
    {
        cvReleaseData(*image);
        cvReleaseImageHeader(image);
    }
@endcode
@param image Double pointer to the image header
*/
CVAPI(void)  cvReleaseImage( IplImage** image );

/** Creates a copy of IPL image (widthStep may differ) */
CVAPI(IplImage*) cvCloneImage( const IplImage* image );

/** @brief Sets the channel of interest in an IplImage.

If the ROI is set to NULL and the coi is *not* 0, the ROI is allocated. Most OpenCV functions do
*not* support the COI setting, so to process an individual image/matrix channel one may copy (via
cvCopy or cvSplit) the channel to a separate image/matrix, process it and then copy the result
back (via cvCopy or cvMerge) if needed.
@param image A pointer to the image header
@param coi The channel of interest. 0 - all channels are selected, 1 - first channel is selected,
etc. Note that the channel indices become 1-based.
 */
CVAPI(void)  cvSetImageCOI( IplImage* image, int coi );

/** @brief Returns the index of the channel of interest.

Returns the channel of interest of in an IplImage. Returned values correspond to the coi in
cvSetImageCOI.
@param image A pointer to the image header
 */
CVAPI(int)  cvGetImageCOI( const IplImage* image );

/** @brief Sets an image Region Of Interest (ROI) for a given rectangle.

If the original image ROI was NULL and the rect is not the whole image, the ROI structure is
allocated.

Most OpenCV functions support the use of ROI and treat the image rectangle as a separate image. For
example, all of the pixel coordinates are counted from the top-left (or bottom-left) corner of the
ROI, not the original image.
@param image A pointer to the image header
@param rect The ROI rectangle
 */
CVAPI(void)  cvSetImageROI( IplImage* image, CvRect rect );

/** @brief Resets the image ROI to include the entire image and releases the ROI structure.

This produces a similar result to the following, but in addition it releases the ROI structure. :
@code
    cvSetImageROI(image, cvRect(0, 0, image->width, image->height ));
    cvSetImageCOI(image, 0);
@endcode
@param image A pointer to the image header
 */
CVAPI(void)  cvResetImageROI( IplImage* image );

/** @brief Returns the image ROI.

If there is no ROI set, cvRect(0,0,image-\>width,image-\>height) is returned.
@param image A pointer to the image header
 */
CVAPI(CvRect) cvGetImageROI( const IplImage* image );

/** @brief Creates a matrix header but does not allocate the matrix data.

The function allocates a new matrix header and returns a pointer to it. The matrix data can then be
allocated using cvCreateData or set explicitly to user-allocated data via cvSetData.
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type Type of the matrix elements, see cvCreateMat
 */
CVAPI(CvMat*)  cvCreateMatHeader( int rows, int cols, int type );

/** @brief Initializes a pre-allocated matrix header.

This function is often used to process raw data with OpenCV matrix functions. For example, the
following code computes the matrix product of two matrices, stored as ordinary arrays:
@code
    double a[] = { 1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12 };

    double b[] = { 1, 5, 9,
                   2, 6, 10,
                   3, 7, 11,
                   4, 8, 12 };

    double c[9];
    CvMat Ma, Mb, Mc ;

    cvInitMatHeader(&Ma, 3, 4, CV_64FC1, a);
    cvInitMatHeader(&Mb, 4, 3, CV_64FC1, b);
    cvInitMatHeader(&Mc, 3, 3, CV_64FC1, c);

    cvMatMulAdd(&Ma, &Mb, 0, &Mc);
    // the c array now contains the product of a (3x4) and b (4x3)
@endcode
@param mat A pointer to the matrix header to be initialized
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type Type of the matrix elements, see cvCreateMat .
@param data Optional: data pointer assigned to the matrix header
@param step Optional: full row width in bytes of the assigned data. By default, the minimal
possible step is used which assumes there are no gaps between subsequent rows of the matrix.
 */
CVAPI(CvMat*) cvInitMatHeader( CvMat* mat, int rows, int cols,
                              int type, void* data CV_DEFAULT(NULL),
                              int step CV_DEFAULT(CV_AUTOSTEP) );

/** @brief Creates a matrix header and allocates the matrix data.

The function call is equivalent to the following code:
@code
    CvMat* mat = cvCreateMatHeader(rows, cols, type);
    cvCreateData(mat);
@endcode
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type The type of the matrix elements in the form
CV_\<bit depth\>\<S|U|F\>C\<number of channels\> , where S=signed, U=unsigned, F=float. For
example, CV _ 8UC1 means the elements are 8-bit unsigned and the there is 1 channel, and CV _
32SC2 means the elements are 32-bit signed and there are 2 channels.
 */
CVAPI(CvMat*)  cvCreateMat( int rows, int cols, int type );

/** @brief Deallocates a matrix.

The function decrements the matrix data reference counter and deallocates matrix header. If the data
reference counter is 0, it also deallocates the data. :
@code
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
@endcode
@param mat Double pointer to the matrix
 */
CVAPI(void)  cvReleaseMat( CvMat** mat );

/** @brief Decrements an array data reference counter.

The function decrements the data reference counter in a CvMat or CvMatND if the reference counter

pointer is not NULL. If the counter reaches zero, the data is deallocated. In the current
implementation the reference counter is not NULL only if the data was allocated using the
cvCreateData function. The counter will be NULL in other cases such as: external data was assigned
to the header using cvSetData, header is part of a larger matrix or image, or the header was
converted from an image or n-dimensional matrix header.
@param arr Pointer to an array header
 */
void  cvDecRefData( CvArr* arr );

/** @brief Increments array data reference counter.

The function increments CvMat or CvMatND data reference counter and returns the new counter value if
the reference counter pointer is not NULL, otherwise it returns zero.
@param arr Array header
 */
int  cvIncRefData( CvArr* arr );

/** Creates an exact copy of the input matrix (except, may be, step value) */
CVAPI(CvMat*) cvCloneMat( const CvMat* mat );


/** @brief Returns matrix header corresponding to the rectangular sub-array of input image or matrix.

The function returns header, corresponding to a specified rectangle of the input array. In other

words, it allows the user to treat a rectangular part of input array as a stand-alone array. ROI is
taken into account by the function so the sub-array of ROI is actually extracted.
@param arr Input array
@param submat Pointer to the resultant sub-array header
@param rect Zero-based coordinates of the rectangle of interest
 */
CVAPI(CvMat*) cvGetSubRect( const CvArr* arr, CvMat* submat, CvRect rect );

/** @brief Returns array row or row span.

The functions return the header, corresponding to a specified row/row span of the input array.
cvGetRow(arr, submat, row) is a shortcut for cvGetRows(arr, submat, row, row+1).
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param start_row Zero-based index of the starting row (inclusive) of the span
@param end_row Zero-based index of the ending row (exclusive) of the span
@param delta_row Index step in the row span. That is, the function extracts every delta_row -th
row from start_row and up to (but not including) end_row .
 */
CVAPI(CvMat*) cvGetRows( const CvArr* arr, CvMat* submat,
                        int start_row, int end_row,
                        int delta_row CV_DEFAULT(1));

/** @overload
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param row Zero-based index of the selected row
*/
CvMat*  cvGetRow( const CvArr* arr, CvMat* submat, int row );

/** @brief Returns one of more array columns.

The functions return the header, corresponding to a specified column span of the input array. That

is, no data is copied. Therefore, any modifications of the submatrix will affect the original array.
If you need to copy the columns, use cvCloneMat. cvGetCol(arr, submat, col) is a shortcut for
cvGetCols(arr, submat, col, col+1).
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param start_col Zero-based index of the starting column (inclusive) of the span
@param end_col Zero-based index of the ending column (exclusive) of the span
 */
CVAPI(CvMat*) cvGetCols( const CvArr* arr, CvMat* submat,
                        int start_col, int end_col );

/** @overload
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param col Zero-based index of the selected column
*/
CvMat*  cvGetCol( const CvArr* arr, CvMat* submat, int col );

/** @brief Returns one of array diagonals.

The function returns the header, corresponding to a specified diagonal of the input array.
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param diag Index of the array diagonal. Zero value corresponds to the main diagonal, -1
corresponds to the diagonal above the main, 1 corresponds to the diagonal below the main, and so
forth.
 */
CVAPI(CvMat*) cvGetDiag( const CvArr* arr, CvMat* submat,
                            int diag CV_DEFAULT(0));

/** low-level scalar <-> raw data conversion functions */
CVAPI(void) cvScalarToRawData( const CvScalar* scalar, void* data, int type,
                              int extend_to_12 CV_DEFAULT(0) );

CVAPI(void) cvRawDataToScalar( const void* data, int type, CvScalar* scalar );

/** @brief Creates a new matrix header but does not allocate the matrix data.

The function allocates a header for a multi-dimensional dense array. The array data can further be
allocated using cvCreateData or set explicitly to user-allocated data via cvSetData.
@param dims Number of array dimensions
@param sizes Array of dimension sizes
@param type Type of array elements, see cvCreateMat
 */
CVAPI(CvMatND*)  cvCreateMatNDHeader( int dims, const int* sizes, int type );

/** @brief Creates the header and allocates the data for a multi-dimensional dense array.

This function call is equivalent to the following code:
@code
    CvMatND* mat = cvCreateMatNDHeader(dims, sizes, type);
    cvCreateData(mat);
@endcode
@param dims Number of array dimensions. This must not exceed CV_MAX_DIM (32 by default, but can be
changed at build time).
@param sizes Array of dimension sizes.
@param type Type of array elements, see cvCreateMat .
 */
CVAPI(CvMatND*)  cvCreateMatND( int dims, const int* sizes, int type );

/** @brief Initializes a pre-allocated multi-dimensional array header.

@param mat A pointer to the array header to be initialized
@param dims The number of array dimensions
@param sizes An array of dimension sizes
@param type Type of array elements, see cvCreateMat
@param data Optional data pointer assigned to the matrix header
 */
CVAPI(CvMatND*)  cvInitMatNDHeader( CvMatND* mat, int dims, const int* sizes,
                                    int type, void* data CV_DEFAULT(NULL) );

/** @brief Deallocates a multi-dimensional array.

The function decrements the array data reference counter and releases the array header. If the
reference counter reaches 0, it also deallocates the data. :
@code
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
@endcode
@param mat Double pointer to the array
 */
void  cvReleaseMatND( CvMatND** mat );

/** Creates a copy of CvMatND (except, may be, steps) */
CVAPI(CvMatND*) cvCloneMatND( const CvMatND* mat );

/** @brief Creates sparse array.

The function allocates a multi-dimensional sparse array. Initially the array contain no elements,
that is PtrND and other related functions will return 0 for every index.
@param dims Number of array dimensions. In contrast to the dense matrix, the number of dimensions is
practically unlimited (up to \f$2^{16}\f$ ).
@param sizes Array of dimension sizes
@param type Type of array elements. The same as for CvMat
 */
CVAPI(CvSparseMat*)  cvCreateSparseMat( int dims, const int* sizes, int type );

/** @brief Deallocates sparse array.

The function releases the sparse array and clears the array pointer upon exit.
@param mat Double pointer to the array
 */
CVAPI(void)  cvReleaseSparseMat( CvSparseMat** mat );

/** Creates a copy of CvSparseMat (except, may be, zero items) */
CVAPI(CvSparseMat*) cvCloneSparseMat( const CvSparseMat* mat );

/** @brief Initializes sparse array elements iterator.

The function initializes iterator of sparse array elements and returns pointer to the first element,
or NULL if the array is empty.
@param mat Input array
@param mat_iterator Initialized iterator
 */
CVAPI(CvSparseNode*) cvInitSparseMatIterator( const CvSparseMat* mat,
                                              CvSparseMatIterator* mat_iterator );

/** @brief Returns the next sparse matrix element

The function moves iterator to the next sparse matrix element and returns pointer to it. In the
current version there is no any particular order of the elements, because they are stored in the
hash table. The sample below demonstrates how to iterate through the sparse matrix:
@code
    // print all the non-zero sparse matrix elements and compute their sum
    double sum = 0;
    int i, dims = cvGetDims(sparsemat);
    CvSparseMatIterator it;
    CvSparseNode* node = cvInitSparseMatIterator(sparsemat, &it);

    for(; node != 0; node = cvGetNextSparseNode(&it))
    {
        int* idx = CV_NODE_IDX(array, node);
        float val = *(float*)CV_NODE_VAL(array, node);
        printf("M");
        for(i = 0; i < dims; i++ )
            printf("[%d]", idx[i]);
        printf("=%g\n", val);

        sum += val;
    }

    printf("nTotal sum = %g\n", sum);
@endcode
@param mat_iterator Sparse array iterator
 */
CvSparseNode* cvGetNextSparseNode( CvSparseMatIterator* mat_iterator );

/** matrix iterator: used for n-ary operations on dense arrays */
typedef struct CvNArrayIterator { ...; } CvNArrayIterator;

/** initializes iterator that traverses through several arrays simulteneously
   (the function together with cvNextArraySlice is used for
    N-ari element-wise operations) */
CVAPI(int) cvInitNArrayIterator( int count, CvArr** arrs,
                                 const CvArr* mask, CvMatND* stubs,
                                 CvNArrayIterator* array_iterator,
                                 int flags CV_DEFAULT(0) );

/** returns zero value if iteration is finished, non-zero (slice length) otherwise */
CVAPI(int) cvNextNArraySlice( CvNArrayIterator* array_iterator );


/** @brief Returns type of array elements.

The function returns type of the array elements. In the case of IplImage the type is converted to
CvMat-like representation. For example, if the image has been created as:
@code
    IplImage* img = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
@endcode
The code cvGetElemType(img) will return CV_8UC3.
@param arr Input array
 */
CVAPI(int) cvGetElemType( const CvArr* arr );

/** @brief Return number of array dimensions

The function returns the array dimensionality and the array of dimension sizes. In the case of
IplImage or CvMat it always returns 2 regardless of number of image/matrix rows. For example, the
following code calculates total number of array elements:
@code
    int sizes[CV_MAX_DIM];
    int i, total = 1;
    int dims = cvGetDims(arr, size);
    for(i = 0; i < dims; i++ )
        total *= sizes[i];
@endcode
@param arr Input array
@param sizes Optional output vector of the array dimension sizes. For 2d arrays the number of rows
(height) goes first, number of columns (width) next.
 */
CVAPI(int) cvGetDims( const CvArr* arr, int* sizes CV_DEFAULT(NULL) );


/** @brief Returns array size along the specified dimension.

@param arr Input array
@param index Zero-based dimension index (for matrices 0 means number of rows, 1 means number of
columns; for images 0 means height, 1 means width)
 */
CVAPI(int) cvGetDimSize( const CvArr* arr, int index );


/** @brief Return pointer to a particular array element.

The functions return a pointer to a specific array element. Number of array dimension should match
to the number of indices passed to the function except for cvPtr1D function that can be used for
sequential access to 1D, 2D or nD dense arrays.

The functions can be used for sparse arrays as well - if the requested node does not exist they
create it and set it to zero.

All these as well as other functions accessing array elements ( cvGetND , cvGetRealND , cvSet
, cvSetND , cvSetRealND ) raise an error in case if the element index is out of range.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param type Optional output parameter: type of matrix elements
 */
CVAPI(uchar*) cvPtr1D( const CvArr* arr, int idx0, int* type CV_DEFAULT(NULL));
/** @overload */
CVAPI(uchar*) cvPtr2D( const CvArr* arr, int idx0, int idx1, int* type CV_DEFAULT(NULL) );
/** @overload */
CVAPI(uchar*) cvPtr3D( const CvArr* arr, int idx0, int idx1, int idx2,
                      int* type CV_DEFAULT(NULL));
/** @overload
@param arr Input array
@param idx Array of the element indices
@param type Optional output parameter: type of matrix elements
@param create_node Optional input parameter for sparse matrices. Non-zero value of the parameter
means that the requested element is created if it does not exist already.
@param precalc_hashval Optional input parameter for sparse matrices. If the pointer is not NULL,
the function does not recalculate the node hash value, but takes it from the specified location.
It is useful for speeding up pair-wise operations (TODO: provide an example)
*/
CVAPI(uchar*) cvPtrND( const CvArr* arr, const int* idx, int* type CV_DEFAULT(NULL),
                      int create_node CV_DEFAULT(1),
                      unsigned* precalc_hashval CV_DEFAULT(NULL));

/** @brief Return a specific array element.

The functions return a specific array element. In the case of a sparse array the functions return 0
if the requested node does not exist (no new node is created by the functions).
@param arr Input array
@param idx0 The first zero-based component of the element index
 */
CVAPI(CvScalar) cvGet1D( const CvArr* arr, int idx0 );
/** @overload */
CVAPI(CvScalar) cvGet2D( const CvArr* arr, int idx0, int idx1 );
/** @overload */
CVAPI(CvScalar) cvGet3D( const CvArr* arr, int idx0, int idx1, int idx2 );
/** @overload
@param arr Input array
@param idx Array of the element indices
*/
CVAPI(CvScalar) cvGetND( const CvArr* arr, const int* idx );

/** @brief Return a specific element of single-channel 1D, 2D, 3D or nD array.

Returns a specific element of a single-channel array. If the array has multiple channels, a runtime
error is raised. Note that Get?D functions can be used safely for both single-channel and
multiple-channel arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new
node is created by the functions).
@param arr Input array. Must have a single channel.
@param idx0 The first zero-based component of the element index
 */
CVAPI(double) cvGetReal1D( const CvArr* arr, int idx0 );
/** @overload */
CVAPI(double) cvGetReal2D( const CvArr* arr, int idx0, int idx1 );
/** @overload */
CVAPI(double) cvGetReal3D( const CvArr* arr, int idx0, int idx1, int idx2 );
/** @overload
@param arr Input array. Must have a single channel.
@param idx Array of the element indices
*/
CVAPI(double) cvGetRealND( const CvArr* arr, const int* idx );

/** @brief Change the particular array element.

The functions assign the new value to a particular array element. In the case of a sparse array the
functions create the node if it does not exist yet.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param value The assigned value
 */
CVAPI(void) cvSet1D( CvArr* arr, int idx0, CvScalar value );
/** @overload */
CVAPI(void) cvSet2D( CvArr* arr, int idx0, int idx1, CvScalar value );
/** @overload */
CVAPI(void) cvSet3D( CvArr* arr, int idx0, int idx1, int idx2, CvScalar value );
/** @overload
@param arr Input array
@param idx Array of the element indices
@param value The assigned value
*/
CVAPI(void) cvSetND( CvArr* arr, const int* idx, CvScalar value );

/** @brief Change a specific array element.

The functions assign a new value to a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that the Set\*D function can be used safely for
both single-channel and multiple-channel arrays, though they are a bit slower.

In the case of a sparse array the functions create the node if it does not yet exist.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param value The assigned value
 */
CVAPI(void) cvSetReal1D( CvArr* arr, int idx0, double value );
/** @overload */
CVAPI(void) cvSetReal2D( CvArr* arr, int idx0, int idx1, double value );
/** @overload */
CVAPI(void) cvSetReal3D( CvArr* arr, int idx0,
                        int idx1, int idx2, double value );
/** @overload
@param arr Input array
@param idx Array of the element indices
@param value The assigned value
*/
CVAPI(void) cvSetRealND( CvArr* arr, const int* idx, double value );

/** clears element of ND dense array,
   in case of sparse arrays it deletes the specified node */
CVAPI(void) cvClearND( CvArr* arr, const int* idx );

/** @brief Returns matrix header for arbitrary array.

The function returns a matrix header for the input array that can be a matrix - CvMat, an image -
IplImage, or a multi-dimensional dense array - CvMatND (the third option is allowed only if
allowND != 0) . In the case of matrix the function simply returns the input pointer. In the case of
IplImage\* or CvMatND it initializes the header structure with parameters of the current image ROI
and returns &header. Because COI is not supported by CvMat, it is returned separately.

The function provides an easy way to handle both types of arrays - IplImage and CvMat using the same
code. Input array must have non-zero data pointer, otherwise the function will report an error.

@note If the input array is IplImage with planar data layout and COI set, the function returns the
pointer to the selected plane and COI == 0. This feature allows user to process IplImage structures
with planar data layout, even though OpenCV does not support such images.
@param arr Input array
@param header Pointer to CvMat structure used as a temporary buffer
@param coi Optional output parameter for storing COI
@param allowND If non-zero, the function accepts multi-dimensional dense arrays (CvMatND\*) and
returns 2D matrix (if CvMatND has two dimensions) or 1D matrix (when CvMatND has 1 dimension or
more than 2 dimensions). The CvMatND array must be continuous.
@sa cvGetImage, cvarrToMat.
 */
CVAPI(CvMat*) cvGetMat( const CvArr* arr, CvMat* header,
                       int* coi CV_DEFAULT(NULL),
                       int allowND CV_DEFAULT(0));

/** @brief Returns image header for arbitrary array.

The function returns the image header for the input array that can be a matrix (CvMat) or image
(IplImage). In the case of an image the function simply returns the input pointer. In the case of
CvMat it initializes an image_header structure with the parameters of the input matrix. Note that
if we transform IplImage to CvMat using cvGetMat and then transform CvMat back to IplImage using
this function, we will get different headers if the ROI is set in the original image.
@param arr Input array
@param image_header Pointer to IplImage structure used as a temporary buffer
 */
CVAPI(IplImage*) cvGetImage( const CvArr* arr, IplImage* image_header );


/** @brief Changes the shape of a multi-dimensional array without copying the data.

The function is an advanced version of cvReshape that can work with multi-dimensional arrays as
well (though it can work with ordinary images and matrices) and change the number of dimensions.

Below are the two samples from the cvReshape description rewritten using cvReshapeMatND:
@code
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    IplImage gray_img_hdr, *gray_img;
    gray_img = (IplImage*)cvReshapeMatND(color_img, sizeof(gray_img_hdr), &gray_img_hdr, 1, 0, 0);
    ...
    int size[] = { 2, 2, 2 };
    CvMatND* mat = cvCreateMatND(3, size, CV_32F);
    CvMat row_header, *row;
    row = (CvMat*)cvReshapeMatND(mat, sizeof(row_header), &row_header, 0, 1, 0);
@endcode
In C, the header file for this function includes a convenient macro cvReshapeND that does away with
the sizeof_header parameter. So, the lines containing the call to cvReshapeMatND in the examples
may be replaced as follow:
@code
    gray_img = (IplImage*)cvReshapeND(color_img, &gray_img_hdr, 1, 0, 0);
    ...
    row = (CvMat*)cvReshapeND(mat, &row_header, 0, 1, 0);
@endcode
@param arr Input array
@param sizeof_header Size of output header to distinguish between IplImage, CvMat and CvMatND
output headers
@param header Output header to be filled
@param new_cn New number of channels. new_cn = 0 means that the number of channels remains
unchanged.
@param new_dims New number of dimensions. new_dims = 0 means that the number of dimensions
remains the same.
@param new_sizes Array of new dimension sizes. Only new_dims-1 values are used, because the
total number of elements must remain the same. Thus, if new_dims = 1, new_sizes array is not
used.
 */
CVAPI(CvArr*) cvReshapeMatND( const CvArr* arr,
                             int sizeof_header, CvArr* header,
                             int new_cn, int new_dims, int* new_sizes );

/** @brief Changes shape of matrix/image without copying data.

The function initializes the CvMat header so that it points to the same data as the original array
but has a different shape - different number of channels, different number of rows, or both.

The following example code creates one image buffer and two image headers, the first is for a
320x240x3 image and the second is for a 960x240x1 image:
@code
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    CvMat gray_mat_hdr;
    IplImage gray_img_hdr, *gray_img;
    cvReshape(color_img, &gray_mat_hdr, 1);
    gray_img = cvGetImage(&gray_mat_hdr, &gray_img_hdr);
@endcode
And the next example converts a 3x3 matrix to a single 1x9 vector:
@code
    CvMat* mat = cvCreateMat(3, 3, CV_32F);
    CvMat row_header, *row;
    row = cvReshape(mat, &row_header, 0, 1);
@endcode
@param arr Input array
@param header Output header to be filled
@param new_cn New number of channels. 'new_cn = 0' means that the number of channels remains
unchanged.
@param new_rows New number of rows. 'new_rows = 0' means that the number of rows remains
unchanged unless it needs to be changed according to new_cn value.
*/
CVAPI(CvMat*) cvReshape( const CvArr* arr, CvMat* header,
                        int new_cn, int new_rows CV_DEFAULT(0) );

/** Repeats source 2d array several times in both horizontal and
   vertical direction to fill destination array */
CVAPI(void) cvRepeat( const CvArr* src, CvArr* dst );

/** @brief Allocates array data

The function allocates image, matrix or multi-dimensional dense array data. Note that in the case of
matrix types OpenCV allocation functions are used. In the case of IplImage they are used unless
CV_TURN_ON_IPL_COMPATIBILITY() has been called before. In the latter case IPL functions are used
to allocate the data.
@param arr Array header
 */
CVAPI(void)  cvCreateData( CvArr* arr );

/** @brief Releases array data.

The function releases the array data. In the case of CvMat or CvMatND it simply calls
cvDecRefData(), that is the function can not deallocate external data. See also the note to
cvCreateData .
@param arr Array header
 */
CVAPI(void)  cvReleaseData( CvArr* arr );

/** @brief Assigns user data to the array header.

The function assigns user data to the array header. Header should be initialized before using
cvCreateMatHeader, cvCreateImageHeader, cvCreateMatNDHeader, cvInitMatHeader,
cvInitImageHeader or cvInitMatNDHeader.
@param arr Array header
@param data User data
@param step Full row length in bytes
 */
CVAPI(void)  cvSetData( CvArr* arr, void* data, int step );

/** @brief Retrieves low-level information about the array.

The function fills output variables with low-level information about the array data. All output

parameters are optional, so some of the pointers may be set to NULL. If the array is IplImage with
ROI set, the parameters of ROI are returned.

The following example shows how to get access to array elements. It computes absolute values of the
array elements :
@code
    float* data;
    int step;
    CvSize size;

    cvGetRawData(array, (uchar**)&data, &step, &size);
    step /= sizeof(data[0]);

    for(int y = 0; y < size.height; y++, data += step )
        for(int x = 0; x < size.width; x++ )
            data[x] = (float)fabs(data[x]);
@endcode
@param arr Array header
@param data Output pointer to the whole image origin or ROI origin if ROI is set
@param step Output full row length in bytes
@param roi_size Output ROI size
 */
CVAPI(void) cvGetRawData( const CvArr* arr, uchar** data,
                         int* step CV_DEFAULT(NULL),
                         CvSize* roi_size CV_DEFAULT(NULL));

/** @brief Returns size of matrix or image ROI.

The function returns number of rows (CvSize::height) and number of columns (CvSize::width) of the
input matrix or image. In the case of image the size of ROI is returned.
@param arr array header
 */
CVAPI(CvSize) cvGetSize( const CvArr* arr );

/** @brief Copies one array to another.

The function copies selected elements from an input array to an output array:

\f[\texttt{dst} (I)= \texttt{src} (I)  \quad \text{if} \quad \texttt{mask} (I)  \ne 0.\f]

If any of the passed arrays is of IplImage type, then its ROI and COI fields are used. Both arrays
must have the same type, the same number of dimensions, and the same size. The function can also
copy sparse arrays (mask is not supported in this case).
@param src The source array
@param dst The destination array
@param mask Operation mask, 8-bit single channel array; specifies elements of the destination array
to be changed
 */
CVAPI(void)  cvCopy( const CvArr* src, CvArr* dst,
                     const CvArr* mask CV_DEFAULT(NULL) );

/** @brief Sets every element of an array to a given value.

The function copies the scalar value to every selected element of the destination array:
\f[\texttt{arr} (I)= \texttt{value} \quad \text{if} \quad \texttt{mask} (I)  \ne 0\f]
If array arr is of IplImage type, then is ROI used, but COI must not be set.
@param arr The destination array
@param value Fill value
@param mask Operation mask, 8-bit single channel array; specifies elements of the destination
array to be changed
 */
CVAPI(void)  cvSet( CvArr* arr, CvScalar value,
                    const CvArr* mask CV_DEFAULT(NULL) );

/** @brief Clears the array.

The function clears the array. In the case of dense arrays (CvMat, CvMatND or IplImage),
cvZero(array) is equivalent to cvSet(array,cvScalarAll(0),0). In the case of sparse arrays all the
elements are removed.
@param arr Array to be cleared
 */
CVAPI(void)  cvSetZero( CvArr* arr );

/** Splits a multi-channel array into the set of single-channel arrays or
   extracts particular [color] plane */
CVAPI(void)  cvSplit( const CvArr* src, CvArr* dst0, CvArr* dst1,
                      CvArr* dst2, CvArr* dst3 );

/** Merges a set of single-channel arrays into the single multi-channel array
   or inserts one particular [color] plane to the array */
CVAPI(void)  cvMerge( const CvArr* src0, const CvArr* src1,
                      const CvArr* src2, const CvArr* src3,
                      CvArr* dst );

/** Copies several channels from input arrays to
   certain channels of output arrays */
CVAPI(void)  cvMixChannels( const CvArr** src, int src_count,
                            CvArr** dst, int dst_count,
                            const int* from_to, int pair_count );

/** @brief Converts one array to another with optional linear transformation.

The function has several different purposes, and thus has several different names. It copies one
array to another with optional scaling, which is performed first, and/or optional type conversion,
performed after:

\f[\texttt{dst} (I) =  \texttt{scale} \texttt{src} (I) + ( \texttt{shift} _0, \texttt{shift} _1,...)\f]

All the channels of multi-channel arrays are processed independently.

The type of conversion is done with rounding and saturation, that is if the result of scaling +
conversion can not be represented exactly by a value of the destination array element type, it is
set to the nearest representable value on the real axis.
@param src Source array
@param dst Destination array
@param scale Scale factor
@param shift Value added to the scaled source array elements
 */
CVAPI(void)  cvConvertScale( const CvArr* src, CvArr* dst,
                             double scale CV_DEFAULT(1),
                             double shift CV_DEFAULT(0) );


/** Performs linear transformation on every source array element,
   stores absolute value of the result:
   dst(x,y,c) = abs(scale*src(x,y,c)+shift).
   destination array must have 8u type.
   In other cases one may use cvConvertScale + cvAbsDiffS */
CVAPI(void)  cvConvertScaleAbs( const CvArr* src, CvArr* dst,
                                double scale CV_DEFAULT(1),
                                double shift CV_DEFAULT(0) );

/** checks termination criteria validity and
   sets eps to default_eps (if it is not set),
   max_iter to default_max_iters (if it is not set)
*/
CVAPI(CvTermCriteria) cvCheckTermCriteria( CvTermCriteria criteria,
                                           double default_eps,
                                           int default_max_iters );

/****************************************************************************************\
*                                   Dynamic Data structures                              *
\****************************************************************************************/

/******************************** Memory storage ****************************************/

/** Creates new memory storage.
   block_size == 0 means that default,
   somewhat optimal size, is used (currently, it is 64K) */
CVAPI(CvMemStorage*)  cvCreateMemStorage( int block_size CV_DEFAULT(0));

/*********************************** Sequence *******************************************/

/** Retrieves pointer to specified sequence element.
   Negative indices are supported and mean counting from the end
   (e.g -1 means the last sequence element) */
CVAPI(schar*)  cvGetSeqElem( const CvSeq* seq, int index );

/************************************* CvScalar *****************************************/
CvScalar  cvScalar( double val0, double val1 CV_DEFAULT(0),
                               double val2 CV_DEFAULT(0), double val3 CV_DEFAULT(0));
CvScalar  cvRealScalar( double val0 );
CvScalar  cvScalarAll( double val0123 );

/** @brief Loads an object from a file.

The function loads an object from a file. It basically reads the specified file, find the first
top-level node and calls cvRead for that node. If the file node does not have type information or
the type information can not be found by the type name, the function returns NULL. After the object
is loaded, the file storage is closed and all the temporary buffers are deleted. Thus, to load a
dynamic structure, such as a sequence, contour, or graph, one should pass a valid memory storage
destination to the function.
@param filename File name
@param memstorage Memory storage for dynamic structures, such as CvSeq or CvGraph . It is not used
for matrices or images.
@param name Optional object name. If it is NULL, the first top-level object in the storage will be
loaded.
@param real_name Optional output parameter that will contain the name of the loaded object
(useful if name=NULL )
 */
CVAPI(void*) cvLoad( const char* filename,
                     CvMemStorage* memstorage CV_DEFAULT(NULL),
                     const char* name CV_DEFAULT(NULL),
                     const char** real_name CV_DEFAULT(NULL) );

/** constructs CvSize structure. */
CvSize  cvSize( int width, int height );

/** constructs CvRect structure. */
CvRect  cvRect( int x, int y, int width, int height );

/** @brief Draws a rectangle given two opposite corners of the rectangle (pt1 & pt2)

   if thickness<0 (e.g. thickness == CV_FILLED), the filled box is drawn
@see cv::rectangle
*/
CVAPI(void)  cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2,
                          CvScalar color, int thickness CV_DEFAULT(1),
                          int line_type CV_DEFAULT(8),
                          int shift CV_DEFAULT(0));

/****************************************************************************************\
*                                Matrix operations                                       *
\****************************************************************************************/

/** @brief Calculates the cross product of two 3D vectors.

The function calculates the cross product of two 3D vectors:
\f[\texttt{dst} =  \texttt{src1} \times \texttt{src2}\f]
or:
\f[\begin{array}{l} \texttt{dst} _1 =  \texttt{src1} _2  \texttt{src2} _3 -  \texttt{src1} _3  \texttt{src2} _2 \\ \texttt{dst} _2 =  \texttt{src1} _3  \texttt{src2} _1 -  \texttt{src1} _1  \texttt{src2} _3 \\ \texttt{dst} _3 =  \texttt{src1} _1  \texttt{src2} _2 -  \texttt{src1} _2  \texttt{src2} _1 \end{array}\f]
@param src1 The first source vector
@param src2 The second source vector
@param dst The destination vector
 */
CVAPI(void)  cvCrossProduct( const CvArr* src1, const CvArr* src2, CvArr* dst );

/** Extended matrix transform:
   dst = alpha*op(A)*op(B) + beta*op(C), where op(X) is X or X^T */
CVAPI(void)  cvGEMM( const CvArr* src1, const CvArr* src2, double alpha,
                     const CvArr* src3, double beta, CvArr* dst,
                     int tABC CV_DEFAULT(0));

/** Transforms each element of source array and stores
   resultant vectors in destination array */
CVAPI(void)  cvTransform( const CvArr* src, CvArr* dst,
                          const CvMat* transmat,
                          const CvMat* shiftvec CV_DEFAULT(NULL));

/** Does perspective transform on every element of input array */
CVAPI(void)  cvPerspectiveTransform( const CvArr* src, CvArr* dst,
                                     const CvMat* mat );

/** Calculates (A-delta)*(A-delta)^T (order=0) or (A-delta)^T*(A-delta) (order=1) */
CVAPI(void) cvMulTransposed( const CvArr* src, CvArr* dst, int order,
                             const CvArr* delta CV_DEFAULT(NULL),
                             double scale CV_DEFAULT(1.) );

/** Tranposes matrix. Square matrices can be transposed in-place */
CVAPI(void)  cvTranspose( const CvArr* src, CvArr* dst );

/** Completes the symmetric matrix from the lower (LtoR=0) or from the upper (LtoR!=0) part */
CVAPI(void)  cvCompleteSymm( CvMat* matrix, int LtoR CV_DEFAULT(0) );

/** Mirror array data around horizontal (flip=0),
   vertical (flip=1) or both(flip=-1) axises:
   cvFlip(src) flips images vertically and sequences horizontally (inplace) */
CVAPI(void)  cvFlip( const CvArr* src, CvArr* dst CV_DEFAULT(NULL),
                     int flip_mode CV_DEFAULT(0));

/** Performs Singular Value Decomposition of a matrix */
CVAPI(void)   cvSVD( CvArr* A, CvArr* W, CvArr* U CV_DEFAULT(NULL),
                     CvArr* V CV_DEFAULT(NULL), int flags CV_DEFAULT(0));

/** Performs Singular Value Back Substitution (solves A*X = B):
   flags must be the same as in cvSVD */
CVAPI(void)   cvSVBkSb( const CvArr* W, const CvArr* U,
                        const CvArr* V, const CvArr* B,
                        CvArr* X, int flags );

/** Inverts matrix */
CVAPI(double)  cvInvert( const CvArr* src, CvArr* dst,
                         int method CV_DEFAULT(CV_LU));

/** Solves linear system (src1)*(dst) = (src2)
   (returns 0 if src1 is a singular and CV_LU method is used) */
CVAPI(int)  cvSolve( const CvArr* src1, const CvArr* src2, CvArr* dst,
                     int method CV_DEFAULT(CV_LU));

/** Calculates determinant of input matrix */
CVAPI(double) cvDet( const CvArr* mat );

/** Calculates trace of the matrix (sum of elements on the main diagonal) */
CVAPI(CvScalar) cvTrace( const CvArr* mat );

/** Finds eigen values and vectors of a symmetric matrix */
CVAPI(void)  cvEigenVV( CvArr* mat, CvArr* evects, CvArr* evals,
                        double eps CV_DEFAULT(0),
                        int lowindex CV_DEFAULT(-1),
                        int highindex CV_DEFAULT(-1));

///* Finds selected eigen values and vectors of a symmetric matrix */
//CVAPI(void)  cvSelectedEigenVV( CvArr* mat, CvArr* evects, CvArr* evals,
//                                int lowindex, int highindex );

/** Makes an identity matrix (mat_ij = i == j) */
CVAPI(void)  cvSetIdentity( CvArr* mat, CvScalar value );

/** Fills matrix with given range of numbers */
CVAPI(CvArr*)  cvRange( CvArr* mat, double start, double end );

/** Calculates covariation matrix for a set of vectors
@see @ref core_c_CovarFlags "flags"
*/
CVAPI(void)  cvCalcCovarMatrix( const CvArr** vects, int count,
                                CvArr* cov_mat, CvArr* avg, int flags );

CVAPI(void)  cvCalcPCA( const CvArr* data, CvArr* mean,
                        CvArr* eigenvals, CvArr* eigenvects, int flags );

CVAPI(void)  cvProjectPCA( const CvArr* data, const CvArr* mean,
                           const CvArr* eigenvects, CvArr* result );

CVAPI(void)  cvBackProjectPCA( const CvArr* proj, const CvArr* mean,
                               const CvArr* eigenvects, CvArr* result );

/** Calculates Mahalanobis(weighted) distance */
CVAPI(double)  cvMahalanobis( const CvArr* vec1, const CvArr* vec2, const CvArr* mat );

/** @brief Warps image with affine transform
@note ::cvGetQuadrangleSubPix is similar to ::cvWarpAffine, but the outliers are extrapolated using
replication border mode.
@see cv::warpAffine
*/
CVAPI(void)  cvWarpAffine( const CvArr* src, CvArr* dst, const CvMat* map_matrix,
                           int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS),
                           CvScalar fillval );

/* "black box" capture structure */
typedef struct CvCapture CvCapture;

/* start capturing frames from video file */
CvCapture* cvCreateFileCapture( const char* filename );

/* start capturing frames from camera: index = camera_index + domain_offset (CV_CAP_*) */
CvCapture* cvCreateCameraCapture( int index );

/* retrieve or set capture properties */
double cvGetCaptureProperty( CvCapture* capture, int property_id );
int    cvSetCaptureProperty( CvCapture* capture, int property_id, double value );

/** Sub-pixel interpolation methods */
enum
{
    CV_INTER_NN        =0,
    CV_INTER_LINEAR    =1,
    CV_INTER_CUBIC     =2,
    CV_INTER_AREA      =3,
    CV_INTER_LANCZOS4  =4
};

/** ... and other image warping flags */
enum
{
    CV_WARP_FILL_OUTLIERS =8,
    CV_WARP_INVERSE_MAP  =16
};

/** Shapes of a structuring element for morphological operations
@see cv::MorphShapes, cv::getStructuringElement
*/
enum MorphShapes_c
{
    CV_SHAPE_RECT      =0,
    CV_SHAPE_CROSS     =1,
    CV_SHAPE_ELLIPSE   =2,
    CV_SHAPE_CUSTOM    =100 //!< custom structuring element
};

/** Morphological operations */
enum
{
    CV_MOP_ERODE        =0,
    CV_MOP_DILATE       =1,
    CV_MOP_OPEN         =2,
    CV_MOP_CLOSE        =3,
    CV_MOP_GRADIENT     =4,
    CV_MOP_TOPHAT       =5,
    CV_MOP_BLACKHAT     =6
};

enum
{
    // modes of the controlling registers (can be: auto, manual, auto single push, absolute Latter allowed with any other mode)
    // every feature can have only one mode turned on at a time
    CV_CAP_PROP_DC1394_OFF         = ...,  //turn the feature off (not controlled manually nor automatically)
    CV_CAP_PROP_DC1394_MODE_MANUAL = ..., //set automatically when a value of the feature is set by the user
    CV_CAP_PROP_DC1394_MODE_AUTO = ...,
    CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = ...,
    CV_CAP_PROP_POS_MSEC       =...,
    CV_CAP_PROP_POS_FRAMES     =...,
    CV_CAP_PROP_POS_AVI_RATIO  =...,
    CV_CAP_PROP_FRAME_WIDTH    =...,
    CV_CAP_PROP_FRAME_HEIGHT   =...,
    CV_CAP_PROP_FPS            =...,
    CV_CAP_PROP_FOURCC         =...,
    CV_CAP_PROP_FRAME_COUNT    =...,
    CV_CAP_PROP_FORMAT         =...,
    CV_CAP_PROP_MODE           =...,
    CV_CAP_PROP_BRIGHTNESS    =...,
    CV_CAP_PROP_CONTRAST      =...,
    CV_CAP_PROP_SATURATION    =...,
    CV_CAP_PROP_HUE           =...,
    CV_CAP_PROP_GAIN          =...,
    CV_CAP_PROP_EXPOSURE      =...,
    CV_CAP_PROP_CONVERT_RGB   =...,
    CV_CAP_PROP_WHITE_BALANCE_BLUE_U =...,
    CV_CAP_PROP_RECTIFICATION =...,
    CV_CAP_PROP_MONOCHROME    =...,
    CV_CAP_PROP_SHARPNESS     =...,
    CV_CAP_PROP_AUTO_EXPOSURE =...,
                                   // user can adjust refernce level
                                   // using this feature
    CV_CAP_PROP_GAMMA         =...,
    CV_CAP_PROP_TEMPERATURE   =...,
    CV_CAP_PROP_TRIGGER       =...,
    CV_CAP_PROP_TRIGGER_DELAY =...,
    CV_CAP_PROP_WHITE_BALANCE_RED_V =...,
    CV_CAP_PROP_ZOOM          =...,
    CV_CAP_PROP_FOCUS         =...,
    CV_CAP_PROP_GUID          =...,
    CV_CAP_PROP_ISO_SPEED     =...,
    CV_CAP_PROP_MAX_DC1394    =...,
    CV_CAP_PROP_BACKLIGHT     =...,
    CV_CAP_PROP_PAN           =...,
    CV_CAP_PROP_TILT          =...,
    CV_CAP_PROP_ROLL          =...,
    CV_CAP_PROP_IRIS          =...,
    CV_CAP_PROP_SETTINGS      =...,
    CV_CAP_PROP_BUFFERSIZE    =...,

    CV_CAP_PROP_AUTOGRAB      =..., // property for videoio class CvCapture_Android only
    CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING=..., // readonly, tricky property, returns cpnst char* indeed
    CV_CAP_PROP_PREVIEW_FORMAT=..., // readonly, tricky property, returns cpnst char* indeed

    // OpenNI map generators
    CV_CAP_OPENNI_DEPTH_GENERATOR =...,
    CV_CAP_OPENNI_IMAGE_GENERATOR =...,
    CV_CAP_OPENNI_GENERATORS_MASK =...,

    // Properties of cameras available through OpenNI interfaces
    CV_CAP_PROP_OPENNI_OUTPUT_MODE     =...,
    CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH =..., // in mm
    CV_CAP_PROP_OPENNI_BASELINE        =..., // in mm
    CV_CAP_PROP_OPENNI_FOCAL_LENGTH    =..., // in pixels
    CV_CAP_PROP_OPENNI_REGISTRATION    =..., // flag
    CV_CAP_PROP_OPENNI_REGISTRATION_ON =..., // flag that synchronizes the remapping depth map to image map
                                                                          // by changing depth generator's view point (if the flag is "on") or
                                                                          // sets this view point to its normal one (if the flag is "off").
    CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC =...,
    CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE   =...,
    CV_CAP_PROP_OPENNI_CIRCLE_BUFFER     =...,
    CV_CAP_PROP_OPENNI_MAX_TIME_DURATION =...,

    CV_CAP_PROP_OPENNI_GENERATOR_PRESENT =...,
    CV_CAP_PROP_OPENNI2_SYNC =...,
    CV_CAP_PROP_OPENNI2_MIRROR =...,

    CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT         =...,
    CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE     =...,
    CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE        =...,
    CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH    =...,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION    =...,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON =...,

    // Properties of cameras available through GStreamer interface
    CV_CAP_GSTREAMER_QUEUE_LENGTH           =..., // default is 1

    // PVAPI
    CV_CAP_PROP_PVAPI_MULTICASTIP           =..., // ip for anable multicast master mode. 0 for disable multicast
    CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE =..., // FrameStartTriggerMode: Determines how a frame is initiated
    CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL  =..., // Horizontal sub-sampling of the image
    CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL    =..., // Vertical sub-sampling of the image
    CV_CAP_PROP_PVAPI_BINNINGX              =..., // Horizontal binning factor
    CV_CAP_PROP_PVAPI_BINNINGY              =..., // Vertical binning factor
    CV_CAP_PROP_PVAPI_PIXELFORMAT           =..., // Pixel format

    // Properties of cameras available through XIMEA SDK interface
    CV_CAP_PROP_XI_DOWNSAMPLING  =...,      // Change image resolution by binning or skipping.
    CV_CAP_PROP_XI_DATA_FORMAT   =...,       // Output data format.
    CV_CAP_PROP_XI_OFFSET_X      =...,      // Horizontal offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_OFFSET_Y      =...,      // Vertical offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_TRG_SOURCE    =...,      // Defines source of trigger.
    CV_CAP_PROP_XI_TRG_SOFTWARE  =...,      // Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
    CV_CAP_PROP_XI_GPI_SELECTOR  =...,      // Selects general purpose input
    CV_CAP_PROP_XI_GPI_MODE      =...,      // Set general purpose input mode
    CV_CAP_PROP_XI_GPI_LEVEL     =...,      // Get general purpose level
    CV_CAP_PROP_XI_GPO_SELECTOR  =...,      // Selects general purpose output
    CV_CAP_PROP_XI_GPO_MODE      =...,      // Set general purpose output mode
    CV_CAP_PROP_XI_LED_SELECTOR  =...,      // Selects camera signalling LED
    CV_CAP_PROP_XI_LED_MODE      =...,      // Define camera signalling LED functionality
    CV_CAP_PROP_XI_MANUAL_WB     =...,      // Calculates White Balance(must be called during acquisition)
    CV_CAP_PROP_XI_AUTO_WB       =...,      // Automatic white balance
    CV_CAP_PROP_XI_AEAG          =...,      // Automatic exposure/gain
    CV_CAP_PROP_XI_EXP_PRIORITY  =...,      // Exposure priority (0.5 - exposure 50%, gain 50%).
    CV_CAP_PROP_XI_AE_MAX_LIMIT  =...,      // Maximum limit of exposure in AEAG procedure
    CV_CAP_PROP_XI_AG_MAX_LIMIT  =...,      // Maximum limit of gain in AEAG procedure
    CV_CAP_PROP_XI_AEAG_LEVEL    =...,       // Average intensity of output signal AEAG should achieve(in %)
    CV_CAP_PROP_XI_TIMEOUT       =...,       // Image capture timeout in milliseconds

    // Properties for Android cameras
    CV_CAP_PROP_ANDROID_FLASH_MODE =...,
    CV_CAP_PROP_ANDROID_FOCUS_MODE =...,
    CV_CAP_PROP_ANDROID_WHITE_BALANCE =...,
    CV_CAP_PROP_ANDROID_ANTIBANDING =...,
    CV_CAP_PROP_ANDROID_FOCAL_LENGTH =...,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR =...,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL =...,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR =...,
    CV_CAP_PROP_ANDROID_EXPOSE_LOCK =...,
    CV_CAP_PROP_ANDROID_WHITEBALANCE_LOCK =...,

    // Properties of cameras available through AVFOUNDATION interface
    CV_CAP_PROP_IOS_DEVICE_FOCUS =...,
    CV_CAP_PROP_IOS_DEVICE_EXPOSURE =...,
    CV_CAP_PROP_IOS_DEVICE_FLASH =...,
    CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE =...,
    CV_CAP_PROP_IOS_DEVICE_TORCH =...,

    // Properties of cameras available through Smartek Giganetix Ethernet Vision interface
    /* --- Vladimir Litvinenko (litvinenko.vladimir@gmail.com) --- */
    CV_CAP_PROP_GIGA_FRAME_OFFSET_X =...,
    CV_CAP_PROP_GIGA_FRAME_OFFSET_Y =...,
    CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX =...,
    CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX =...,
    CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH =...,
    CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH =...,

    CV_CAP_PROP_INTELPERC_PROFILE_COUNT               =...,
    CV_CAP_PROP_INTELPERC_PROFILE_IDX                 =...,
    CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE  =...,
    CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE      =...,
    CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD  =...,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ     =...,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT     =...,

    // Intel PerC streams
    CV_CAP_INTELPERC_DEPTH_GENERATOR =...,
    CV_CAP_INTELPERC_IMAGE_GENERATOR =...,
    CV_CAP_INTELPERC_GENERATORS_MASK =...
};

typedef IplImage* (* Cv_iplCreateImageHeader)
                            (int,int,int,char*,char*,int,int,int,int,int,
                            IplROI*,IplImage*,void*,IplTileInfo*);
typedef void (* Cv_iplAllocateImageData)(IplImage*,int,int);
typedef void (* Cv_iplDeallocate)(IplImage*,int);
typedef IplROI* (* Cv_iplCreateROI)(int,int,int,int,int);
typedef IplImage* (* Cv_iplCloneImage)(const IplImage*);

/* Just a combination of cvGrabFrame and cvRetrieveFrame
!!!DO NOT RELEASE or MODIFY the retrieved frame!!!      */
IplImage* cvQueryFrame( CvCapture* capture );

/* "black box" video file writer structure */
typedef struct CvVideoWriter CvVideoWriter;

/* initialize video file writer */
CvVideoWriter* cvCreateVideoWriter( const char* filename, int fourcc,
                                           double fps, CvSize frame_size,
                                           int is_color );

/* write frame to video file */
CVAPI(int) cvWriteFrame( CvVideoWriter* writer, const IplImage* image );

/* close video file writer */
CVAPI(void) cvReleaseVideoWriter( CvVideoWriter** writer );

/* create window */
CVAPI(int) cvNamedWindow( const char* name, int flags CV_DEFAULT(CV_WINDOW_AUTOSIZE) );

/* Set and Get Property of the window */
CVAPI(void) cvSetWindowProperty(const char* name, int prop_id, double prop_value);
CVAPI(double) cvGetWindowProperty(const char* name, int prop_id);

/* display image within window (highgui windows remember their content) */
CVAPI(void) cvShowImage( const char* name, const CvArr* image );

/* resize/move window */
CVAPI(void) cvResizeWindow( const char* name, int width, int height );
CVAPI(void) cvMoveWindow( const char* name, int x, int y );


/* destroy window and all the trackers associated with it */
CVAPI(void) cvDestroyWindow( const char* name );

CVAPI(void) cvDestroyAllWindows(void);

/* get native window handle (HWND in case of Win32 and Widget in case of X Window) */
CVAPI(void*) cvGetWindowHandle( const char* name );

/* get name of highgui window given its native handle */
CVAPI(const char*) cvGetWindowName( void* window_handle );

/* wait for key event infinitely (delay<=0) or for "delay" milliseconds */
CVAPI(int) cvWaitKey(int delay CV_DEFAULT(0));

enum { CV_WINDOW_AUTOSIZE=... };


/****************************************************************************************\
*                         Haar-like Object Detection functions                           *
\****************************************************************************************/


typedef struct CvHaarClassifierCascade { ...;  } CvHaarClassifierCascade;

/* Loads haar classifier cascade from a directory.
   It is obsolete: convert your cascade to xml and use cvLoad instead */
CVAPI(CvHaarClassifierCascade*) cvLoadHaarClassifierCascade(
                    const char* directory, CvSize orig_window_size);

CVAPI(void) cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** cascade );

CVAPI(CvSeq*) cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     double scale_factor CV_DEFAULT(1.1),
                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
                     CvSize min_size , CvSize max_size );

/* sets images for haar classifier cascade */
CVAPI(void) cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                                const CvArr* sum, const CvArr* sqsum,
                                                const CvArr* tilted_sum, double scale );

/* runs the cascade on the specified window */
CVAPI(int) cvRunHaarClassifierCascade( const CvHaarClassifierCascade* cascade,
                                       CvPoint pt, int start_stage CV_DEFAULT(0));
    """,
        ),
    ),
)


if __name__ == "__main__":
    ffi.compile()
