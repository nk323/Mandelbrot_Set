/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * NAME: Neha Kadam
 * ECE 6122 Fall 2015 
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"

#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512
#define NUM_THREADS 	      32
#define IMG_SIZE 	      WINDOW_DIM * WINDOW_DIM
#define NUM_BLOCKS 	      IMG_SIZE/ NUM_THREADS
#define DEBUG 		      0

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
const int maxIt = 2000; // Maximum Iterations

Complex* c = new Complex[IMG_SIZE];//array to hold unique c values
int num_of_iterations[IMG_SIZE];

Complex* dev_c;  //c value array on the device
int* dev_iterations;

bool cudaMode = true;		
bool isSqrDrawn = false;
float dx, dy, diff;	//displacement variables
static int zoomLevel = 0;

// Function Declarations
void init();
void InitializeColors();
void drawMBSet();
void display();
void displayMBSet();

// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors()
{
  srand48(10);
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 6)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
	  colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}

// Class to keep track of mouse click point
class Point
{
public:
  float x;
  float y;

  Point():x(0.0f), y(0.0f){}
};

Point start, end;

// Stack to store old values of minC and maxC when zooming in
stack< pair<Complex,Complex> > memStack;

// CUDA Function to compute MBSet 
__global__ void calcMB(Complex* dev_minC, Complex* dev_maxC, Complex* dev_c, int* dev_iterations)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int x = id / WINDOW_DIM;
  int y = id % WINDOW_DIM;	

  double diffR = dev_maxC->r - dev_minC->r; //get diff between maxC and minC for real and imag
  double diffI = dev_maxC->i - dev_minC->i;
  double idR = (double) x / (WINDOW_DIM - 1); // to generate a unique c use the pixel location 
  double idI = (double) y / (WINDOW_DIM - 1);
  Complex newC = Complex(idR * diffR, idI * diffI);
  dev_c[id] = *dev_minC + newC;

  Complex Z(dev_c[id]);
  dev_iterations[id] = 0;
  
  // now compute Z till either iterations > maxIt or magnitude square > 4
  while(Z.magnitude2() < 4.0 && dev_iterations[id] < maxIt)
  {
    Z = (Z*Z) + dev_c[id];
    dev_iterations[id]++;
  }   

}
// Function to compute the Mandelbrot Set and act as wrapper for calling CUDA function
void drawMBSet()
{
  if(cudaMode)
  {
    if(DEBUG) cout << "Running in CUDA mode\n";
    // malloc
    cudaMalloc((void**)&dev_minC, sizeof(Complex));
    cudaMalloc((void**)&dev_maxC, sizeof(Complex));
    cudaMalloc((void**)&dev_c, IMG_SIZE * sizeof(Complex));
    cudaMalloc((void**)&dev_iterations, IMG_SIZE * sizeof(int));

    //now copy from host to device
    cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, IMG_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iterations, num_of_iterations, IMG_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // call CUDA function
    calcMB<<< NUM_BLOCKS, NUM_THREADS >>>(dev_minC, dev_maxC, dev_c, dev_iterations);

    //now copy resuls from device to host
    cudaMemcpy(c,dev_c, IMG_SIZE * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_of_iterations,dev_iterations, IMG_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  }

  else // calculate without CUDA
  {
    int index = 0;

    for(int i = 0; i < WINDOW_DIM; i++)
    {
      for(int j = 0; j < WINDOW_DIM; j++)
      {
	index = i*WINDOW_DIM + j;
	
	double diffR = maxC.r - minC.r; //get diff between maxC and minC for real an imag
	double diffI = maxC.i - minC.i;
	double idR = (double) i / (WINDOW_DIM - 1); //generate unique c using pixel location 
	double idI = (double) j / (WINDOW_DIM - 1);
	Complex newC = Complex(idR * diffR, idI * diffI);
	c[index] = minC + newC;

	Complex Z(c[index]); // init Z0
	num_of_iterations[index] = 0;  // init number of iterations
		
	// now compute Z till either iterations > maxIt or magnitude sqaure > 4
	while(Z.magnitude2() < 4.0 && num_of_iterations[index] < maxIt)
	{
	  Z = (Z*Z) + c[index];
	  num_of_iterations[index]++;

	}
      }
    }

  }

}

void display()
{

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); 
  displayMBSet();

  if(isSqrDrawn)
  {
    glColor3f(1.0, 0.0, 0.0);	//Red square
    
    glBegin(GL_LINE_LOOP);
      glVertex2f(start.x, start.y);	// x1,y1
      glVertex2f(end.x, start.y);	// x2,y1
      glVertex2f(end.x, end.y);		// x2,y2
      glVertex2f(start.x, end.y);	// x1,y2
    glEnd();
  }

  glutSwapBuffers();
  
}

void displayMBSet()
{

  int index = 0;
  glBegin(GL_POINTS);

  for(int i = 0; i < WINDOW_DIM; i++)
  {
    for(int j = 0; j < WINDOW_DIM; j++)
    {
	index = i*WINDOW_DIM + j;

	double r = colors[num_of_iterations[index]].r;	
	double g = colors[num_of_iterations[index]].g;		
	double b = colors[num_of_iterations[index]].b;	
	glColor3f(r,g,b);
	glVertex2i(i,j);

    }
  }

  glEnd();

}

void mouse(int button, int state, int x, int y)
{
  
  int i,j;
  
  if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
  {
      start.x = x;
      end.x = x;  
      
      start.y = y;
      end.y = y;
      
      isSqrDrawn = true;
  }

  if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
  {
      // push current minC and maxC values on stack
      memStack.push(make_pair(minC,maxC));

      //now calculate new minC and maxC based on zoom area
    	if(x > start.x)
    	{
      	  if(y > start.y)		// top left to bottom right
      	  {
		end.x = start.x + diff;
		end.y = start.y + diff;
		// if (x1,y1) is start and (x2,y2) is end then
		// new minC at top left i.e. x1, y1
		i = start.x; 
		j = start.y;
		minC = c[i*WINDOW_DIM + j];
		//new maxC at bottom right x2, y2
		i = end.x;
		j = end.y;
		maxC = c[i*WINDOW_DIM + j];
      	  }
      	  else if(y < start.y)		// bottom left to top right
          {
		end.x = start.x + diff;
		end.y = start.y - diff;
		//new minC at top left of box i.e. x1,y2 
		i = start.x;
		j = end.y;
		minC = c[i*WINDOW_DIM + j];
		//new maxC at bottom right of box i.e. x2,y1
		i = end.x;
		j = start.y;
		maxC = c[i*WINDOW_DIM + j];   	
	  }
    	}
    
	else if(x < start.x)
    	{
      	  if(y > start.y)		// top right to bottom left
      	  {
		end.x = start.x - diff;
		end.y = start.y + diff;

		i = end.x;
		j = start.y;
		minC = c[i*WINDOW_DIM + j];

		i = start.x;
		j = end.y;
		maxC = c[i*WINDOW_DIM + j];     	
      	  }
      	  else if (y < start.y)		// bottom right to top left
      	  {
		end.x = start.x - diff;
		end.y = start.y - diff;
 		
		i = end.x;
		j = end.y;
		minC = c[i*WINDOW_DIM + j];

		i = start.x;
		j = start.y;
		maxC = c[i*WINDOW_DIM + j];  	    
	  } 

    	} 
	
	drawMBSet();
	cout << "Zooming in. Level " << ++zoomLevel << endl;
	isSqrDrawn = false;
	glutPostRedisplay();    
   
  }
}

void motion(int x, int y)
{
  
  dx = abs(x - start.x);
  dy = abs(y - start.y);
 
  // Choosing smaller of the two diffs to get side of square
  if(dy > dx) 
    diff = dx;
  else 
    diff = dy;
  
  // Calculating new 'end' co-ordinates based on this diff
  if(x > start.x)
  {
    if(y > start.y)  		// top left to bottom right
    {
	end.x = start.x + diff;
	end.y = start.y + diff;
    }
    else if(y < start.y)	// bottom left to top right
    {
	end.x = start.x + diff;
	end.y = start.y - diff;
    }
  }
  else if(x < start.x)
  {
    if(y > start.y)		// top right to bottom left
    {
	end.x = start.x - diff;
	end.y = start.y + diff;
    }
    else if (y < start.y)	// bottom right to top left
    {
	end.x = start.x - diff;
	end.y = start.y - diff;
    }

  }

  glutPostRedisplay();
 
}

void keyboard(unsigned char key, int x, int y)
{
  if(key == 'B' || key == 'b')
  {
    if(!memStack.empty())
    {
	cout << "Zooming out. Level " << --zoomLevel << endl;

	minC = memStack.top().first;
	maxC = memStack.top().second;
	memStack.pop();	

	//recalculate MBSet with previous values of minC and maxC
	drawMBSet();
	glutPostRedisplay();
    }
    else
    {
	cout << "Already at maximum zoom out" << endl;
    }
  }

}

__host__ void clean()
{
  free(c);
  cudaFree(dev_minC);
  cudaFree(dev_maxC);
  cudaFree(dev_c);
  cudaFree(dev_iterations);
}

void init()
{
  glClearColor(0, 0, 0, 0);
  glViewport(0, 0, WINDOW_DIM, WINDOW_DIM);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, -WINDOW_DIM, WINDOW_DIM);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

int main(int argc, char** argv)
{
  // Initialize OPENGL here
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels
  atexit(clean);

  glutInit(&argc, argv);
  InitializeColors();	// calculate color array based on iteration counts
  drawMBSet();		//calculate MBSet

  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
  glutInitWindowPosition(200, 200);
  glutCreateWindow("Mandelbrot Set");

  init();  
  glutDisplayFunc(display);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);
  
  glutMainLoop(); // THis will callback the display, keyboard and mouse

  return 0;
  
}
