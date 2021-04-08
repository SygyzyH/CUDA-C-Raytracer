#include <tchar.h>
#include <windows.h>
#include <iostream>
#include <inttypes.h>
#include <string>
#include <sstream>
#include "settings.h"
#include "ManagerAPIFunc.h"
#include "handler.h"
// to compile & execute: nvcc main.cu view.cpp manager.cpp -o view -lgdi32 -luser32 && view.exe


LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE: {
            DestroyWindow(hwnd);
            break;
        }
        case WM_DESTROY: {
            PostQuitMessage(0);
            break;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            // call raytracer entry point
            // threading could remove this requirement and instead get the pointer to a buffer
            uint32_t *arr = ManagerGetPixelData();

            // blit array
            HBITMAP map = CreateBitmap(WIDTH, HEIGHT, 1, 32, (void *) arr);
            HDC src = CreateCompatibleDC(hdc);

            SelectObject(src, map);
            BitBlt(hdc, 0, 0, WIDTH, HEIGHT, src, 0, 0, SRCCOPY);

            ManagerFreePixelData();

            DeleteObject(map);
            DeleteDC(src);
            EndPaint(hwnd, &ps);
            break;
        }
        case WM_KEYDOWN: {
            switch (wParam) {
                case VK_ESCAPE: 
                    if (MessageBox(hwnd, _T("Are you sure you would like to quit?"), _T("CUDA RT"), MB_OKCANCEL) == IDOK) 
                        DestroyWindow(hwnd);
                    break;
                case 'a':
                case 'A':
                case VK_LEFT: 
                    activeKeys |= LEFT;
                    break;
                case 'd':
                case 'D':
                case VK_RIGHT: 
                    activeKeys |= RIGHT;
                    break;
                case 'w':
                case 'W':
                case VK_UP: 
                    activeKeys |= FORWARD;
                    break;
                case 's':
                case 'S':
                case VK_DOWN: 
                    activeKeys |= BACKWORD;
                    break;
                case VK_SPACE: 
                    activeKeys |= UP;
                    break;
                case VK_CONTROL:
                    activeKeys |= DOWN;
                    break;
            }
            break;
        }
        case WM_KEYUP: {
            switch (wParam) {
                case 'a':
                case 'A':
                case VK_LEFT:
                    activeKeys &= ~LEFT;
                    break;
                case 'd':
                case 'D':
                case VK_RIGHT:
                    activeKeys &= ~RIGHT;
                    break;
                case 'w':
                case 'W':
                case VK_UP:
                    activeKeys &= ~FORWARD;
                    break;
                case 's':
                case 'S':
                case VK_DOWN:
                    activeKeys &= ~BACKWORD;
                    break;
                case VK_SPACE:
                    activeKeys &= ~UP;
                    break;
                case VK_CONTROL:
                    activeKeys &= ~DOWN;
                    break; 
            }
        }
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    WNDCLASSEX wc;
    HWND hwnd;
    MSG Msg;

    // registering the Window Class
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = 0;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = _T("myWindowClass");
    wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if(!RegisterClassEx(&wc)) {
        MessageBox(NULL, _T("Window Registration Failed"), _T("Error"), MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // creating the Window
    hwnd = CreateWindowEx(WS_EX_CLIENTEDGE, _T("myWindowClass"), _T("CUDA Accelerated Raytracer"), 
                          WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
                          CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, HEIGHT, NULL, NULL, hInstance, NULL);

    if(hwnd == NULL) {
        MessageBox(NULL, _T("Window Creation Failed"), _T("Error"), MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // init the manager and the raytracer
    ManagerInit();

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // message loop
    while(GetMessage(&Msg, NULL, 0, 0) > 0) {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
        // TODO: this should not be here. a threaded input handler should be used instead.
        if (activeKeys != 0) {
            float x = 0, y = 0, z = 0;
            handleKeys(&x, &y, &z);
            ManagerTranslateCamera(x, y, z);
            // not sure which one is faster...
            InvalidateRect(hwnd, NULL, NULL);
            //RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
        }
    }

    // before leaving, cleanup memory used by the manager and raytracer
    ManagerCleanup();

    return Msg.wParam;
}