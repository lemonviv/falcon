//
// Created by wuyuncheng on 5/3/21.
//

#ifndef FALCON_INCLUDE_FALCON_UTILS_BASE64_H_
#define FALCON_INCLUDE_FALCON_UTILS_BASE64_H_

#ifndef _BASE64_H_
#define _BASE64_H_

#include <vector>
#include <string>
typedef unsigned char BYTE;

std::string base64_encode(BYTE const* buf, unsigned int bufLen);
std::vector<BYTE> base64_decode(std::string const&);

#endif

#endif //FALCON_INCLUDE_FALCON_UTILS_BASE64_H_