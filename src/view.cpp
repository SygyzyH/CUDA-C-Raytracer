#include <tchar.h>
#include <windows.h>
#include <iostream>
#include <inttypes.h>
#include "settings.h"
#include "RTAPIFunc.h"
// to compile & execute: nvcc main.cu view.cpp -o view -lgdi32 -luser32 && view.exe


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

            /*uint32_t *arr = (uint32_t *) calloc(HEIGHT * WIDTH, sizeof(uint32_t));
            for (int x = 0; x < WIDTH; x++)
                for (int y = 0; y < HEIGHT; y++)
                    // we gaming 
                    arr[y * WIDTH + x] = 127 + ((int) ((float) x / WIDTH * 256.0f) << 8) + ((int) ((float) y / HEIGHT * 256.0f) << 16);*/

            // call raytracer entry point
            uint32_t *arr = RTEntryPoint();

            // blit array
            HBITMAP map = CreateBitmap(WIDTH, HEIGHT, 1, 32, (void *) arr);
            HDC src = CreateCompatibleDC(hdc);

            SelectObject(src, map);
            BitBlt(hdc, 0, 0, WIDTH, HEIGHT, src, 0, 0, SRCCOPY);

            DeleteObject(map);
            DeleteDC(src);
            EndPaint(hwnd, &ps);
            break;
        }
        case WM_KEYDOWN: {
            switch (wParam) {
                case VK_ESCAPE: {
                    if (MessageBox(hwnd, _T("Are you sure you would like to quit?"), _T("CUDA RT"), MB_OKCANCEL) == IDOK) 
                        DestroyWindow(hwnd);
                    break;
                }
                case VK_LEFT: {
                    RTTranslateCamera(0.0f, 0.0f, 0.1f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
                case VK_RIGHT: {
                    RTTranslateCamera(0.0f, 0.0f, -0.1f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
                case VK_UP: {
                    RTTranslateCamera(0.1f, 0.0f, 0.0f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
                case VK_DOWN: {
                    RTTranslateCamera(-0.1f, 0.0f, -0.0f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
                case VK_SPACE: {
                    RTTranslateCamera(0.0f, 0.1f, 0.0f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
                case VK_CONTROL: {
                    RTTranslateCamera(0.0f, -0.1f, 0.0f);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);
                    break;
                }
            }
            break;
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

    // init the raytracer
    RTInit();

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // message loop
    while(GetMessage(&Msg, NULL, 0, 0) > 0) {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    // before leaving, cleanup memory used by the raytracer
    RTCleanup();

    return Msg.wParam;
}