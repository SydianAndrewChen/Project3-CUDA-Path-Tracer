#pragma once

#include <vector>
#include "depScene.h"
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(HostScene *scene, Scene * pa);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtraceInitBeforeMainLoop(Scene * pa);
void pathtraceFreeAfterMainLoop(Scene * pa);
