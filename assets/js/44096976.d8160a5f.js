(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[3273],{3905:function(n,e,t){"use strict";t.d(e,{Zo:function(){return c},kt:function(){return N}});var r=t(67294);function o(n,e,t){return e in n?Object.defineProperty(n,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):n[e]=t,n}function a(n,e){var t=Object.keys(n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(n);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(n,e).enumerable}))),t.push.apply(t,r)}return t}function i(n){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?a(Object(t),!0).forEach((function(e){o(n,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(n,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(e){Object.defineProperty(n,e,Object.getOwnPropertyDescriptor(t,e))}))}return n}function p(n,e){if(null==n)return{};var t,r,o=function(n,e){if(null==n)return{};var t,r,o={},a=Object.keys(n);for(r=0;r<a.length;r++)t=a[r],e.indexOf(t)>=0||(o[t]=n[t]);return o}(n,e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(n);for(r=0;r<a.length;r++)t=a[r],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(n,t)&&(o[t]=n[t])}return o}var _=r.createContext({}),l=function(n){var e=r.useContext(_),t=e;return n&&(t="function"==typeof n?n(e):i(i({},e),n)),t},c=function(n){var e=l(n.components);return r.createElement(_.Provider,{value:e},n.children)},s={inlineCode:"code",wrapper:function(n){var e=n.children;return r.createElement(r.Fragment,{},e)}},m=r.forwardRef((function(n,e){var t=n.components,o=n.mdxType,a=n.originalType,_=n.parentName,c=p(n,["components","mdxType","originalType","parentName"]),m=l(t),N=o,u=m["".concat(_,".").concat(N)]||m[N]||s[N]||a;return t?r.createElement(u,i(i({ref:e},c),{},{components:t})):r.createElement(u,i({ref:e},c))}));function N(n,e){var t=arguments,o=e&&e.mdxType;if("string"==typeof n||o){var a=t.length,i=new Array(a);i[0]=m;var p={};for(var _ in e)hasOwnProperty.call(e,_)&&(p[_]=e[_]);p.originalType=n,p.mdxType="string"==typeof n?n:o,i[1]=p;for(var l=2;l<a;l++)i[l]=t[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},8312:function(n,e,t){"use strict";t.r(e),t.d(e,{frontMatter:function(){return p},contentTitle:function(){return _},metadata:function(){return l},toc:function(){return c},default:function(){return m}});var r=t(22122),o=t(19756),a=(t(67294),t(3905)),i=["components"],p={},_="\u6df7\u5408\u7f16\u7a0b (C + MicroPython)",l={unversionedId:"chips/micropython/micropython_app_dev/advance/mixed_program",id:"chips/micropython/micropython_app_dev/advance/mixed_program",isDocsHomePage:!1,title:"\u6df7\u5408\u7f16\u7a0b (C + MicroPython)",description:"\u53ef\u80fd\u5b58\u5728\u4e00\u4e9b\u60c5\u51b5\u4e0b\uff0c\u4f60\u5e76\u4e0d\u80fd\u5b8c\u5b8c\u5168\u5168\u8131\u79bb\u539f\u6709\u9879\u76ee\u7684 C \u4ee3\u7801\u7684\u4e1a\u52a1\u903b\u8f91\uff0c\u90a3\u4e48\u4f60\u53ef\u4ee5\u53c2\u8003\u672c\u7ae0\u5bf9\u7f16\u8bd1\u914d\u7f6e\u8fdb\u884c\u4e00\u4e9b\u6539\u52a8\u6765\u4f7f\u7528 MicroPython\u3002",source:"@site/docs/chips/micropython/micropython_app_dev/advance/mixed_program.md",sourceDirName:"chips/micropython/micropython_app_dev/advance",slug:"/chips/micropython/micropython_app_dev/advance/mixed_program",permalink:"/chips/micropython/micropython_app_dev/advance/mixed_program",version:"current",frontMatter:{},sidebar:"micropython",previous:{title:"\u5c4f\u5e55\u663e\u793a",permalink:"/chips/micropython/micropython_app_dev/advance/display"},next:{title:"\u6587\u4ef6\u7cfb\u7edf\u76f8\u5173",permalink:"/chips/micropython/micropython_app_dev/faq/file_system"}},c=[{value:"1. \u7f16\u8bd1\u914d\u7f6e",id:"1-\u7f16\u8bd1\u914d\u7f6e",children:[{value:"\u9879\u76ee\u914d\u7f6e",id:"\u9879\u76ee\u914d\u7f6e",children:[]},{value:"\u677f\u578b\u914d\u7f6e",id:"\u677f\u578b\u914d\u7f6e",children:[]}]},{value:"2. \u6587\u4ef6\u7cfb\u7edf\u914d\u7f6e",id:"2-\u6587\u4ef6\u7cfb\u7edf\u914d\u7f6e",children:[]},{value:"3. \u542f\u52a8 MicroPython",id:"3-\u542f\u52a8-micropython",children:[]}],s={toc:c};function m(n){var e=n.components,t=(0,o.Z)(n,i);return(0,a.kt)("wrapper",(0,r.Z)({},s,t,{components:e,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u6df7\u5408\u7f16\u7a0b-c--micropython"},"\u6df7\u5408\u7f16\u7a0b (C + MicroPython)"),(0,a.kt)("p",null,"\u53ef\u80fd\u5b58\u5728\u4e00\u4e9b\u60c5\u51b5\u4e0b\uff0c\u4f60\u5e76\u4e0d\u80fd\u5b8c\u5b8c\u5168\u5168\u8131\u79bb\u539f\u6709\u9879\u76ee\u7684 C \u4ee3\u7801\u7684\u4e1a\u52a1\u903b\u8f91\uff0c\u90a3\u4e48\u4f60\u53ef\u4ee5\u53c2\u8003\u672c\u7ae0\u5bf9\u7f16\u8bd1\u914d\u7f6e\u8fdb\u884c\u4e00\u4e9b\u6539\u52a8\u6765\u4f7f\u7528 MicroPython\u3002"),(0,a.kt)("h2",{id:"1-\u7f16\u8bd1\u914d\u7f6e"},"1. \u7f16\u8bd1\u914d\u7f6e"),(0,a.kt)("h3",{id:"\u9879\u76ee\u914d\u7f6e"},"\u9879\u76ee\u914d\u7f6e"),(0,a.kt)("p",null,"\u5728\u4f60\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"CMakeLists.txt")," \u4e2d\uff0c\u52a0\u5165"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-cmake"},"include($ENV{MICROPY_SDK}/ports/zephyr/lib_micropython.cmake)\n\ntarget_sources(app PRIVATE\n    # ... your C files ...\n    ${MPY_APP_SRC}\n)\n\ntarget_include_directories(app PRIVATE ${MPY_APP_INCLUDE})\n\ntarget_link_libraries(app PRIVATE ${MICROPY_TARGET})\n")),(0,a.kt)("h3",{id:"\u677f\u578b\u914d\u7f6e"},"\u677f\u578b\u914d\u7f6e"),(0,a.kt)("p",null,"\u5728\u4f60\u9879\u76ee\u4e0b\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"KConfig")," \u6587\u4ef6\u4e2d\uff08\u5982\u679c\u6ca1\u6709\u5c31\u521b\u5efa\u4e00\u4e2a\uff09\uff0c\u52a0\u5165"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-Kconfig"},'source "$MICROPY_SDK/ports/zephyr/Kconfig"\n')),(0,a.kt)("p",null,"\u5bf9\u4f60\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"${board}.conf")," \u6216 ",(0,a.kt)("inlineCode",{parentName:"p"},"prj.conf")," \u6587\u4ef6\u6309\u9700\u589e\u52a0\u4e00\u4e9b\u914d\u7f6e"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-Kconfig"},'# \u5fc5\u9009\u3002\u6587\u4ef6\u7cfb\u7edf\u3002\u4f46 LFS \u6216 FAT \u65e0\u9650\u5236\u3002\nCONFIG_FILE_SYSTEM=y\nCONFIG_FILE_SYSTEM_LITTLEFS=y\nCONFIG_MICROPY_VFS_FAT=n\nCONFIG_MICROPY_VFS_LFS1=y\nCONFIG_MICROPY_VFS_LFS2=y\nCONFIG_FLASH=y\nCONFIG_FLASH_MAP=y\nCONFIG_MPU_ALLOW_FLASH_WRITE=y\nCONFIG_FLASH_PAGE_LAYOUT=y\nCONFIG_DISK_ACCESS=y\n\n# \u53ef\u9009\u3002LVGL GUI \u5e93\u3002\u8981\u6ce8\u610f\uff0c\u663e\u793a\u548c\u89e6\u63a7\u9700\u8981\u81ea\u5df1\u914d\u7f6e\u5bf9\u5e94\u7684\u9879\u4ee5\u53ca\u8bbe\u5907\u6811\u3002\nCONFIG_MPY_LVGL=y\nCONFIG_LV_MEM_CUSTOM=y\nCONFIG_LV_Z_DISPLAY_DEV_NAME="DISPLAY"\nCONFIG_LV_COLOR_DEPTH_16=y\nCONFIG_LV_Z_USE_FILESYSTEM=y\nCONFIG_LV_Z_POINTER_KSCAN=y\nCONFIG_LV_Z_POINTER_KSCAN_DEV_NAME="KSCAN"\nCONFIG_LV_Z_HOR_RES_MAX=240\nCONFIG_LV_Z_VER_RES_MAX=320\nCONFIG_LV_Z_DOUBLE_VDB=y\nCONFIG_LV_Z_MEM_POOL_MAX_SIZE=81920\n# CONFIG_LV_USE_FREETYPE=y\n\n# \u53ef\u9009\u3002\u8054\u7f51\u80fd\u529b\n# Networking config\nCONFIG_NETWORKING=y\nCONFIG_NET_IPV4=y\nCONFIG_NET_IPV6=y\nCONFIG_NET_UDP=y\nCONFIG_NET_TCP=y\nCONFIG_NET_SOCKETS=y\nCONFIG_NET_SOCKETS_POSIX_NAMES=n\nCONFIG_TEST_RANDOM_GENERATOR=y\nCONFIG_NET_CONFIG_SETTINGS=y\nCONFIG_NET_CONFIG_INIT_TIMEOUT=3\nCONFIG_NET_CONFIG_NEED_IPV6=y\nCONFIG_NET_CONFIG_NEED_IPV4=y\n# DNS\nCONFIG_DNS_RESOLVER=y\nCONFIG_DNS_RESOLVER_ADDITIONAL_QUERIES=2\nCONFIG_DNS_SERVER_IP_ADDRESSES=y\n# Static IP addresses\nCONFIG_NET_CONFIG_MY_IPV6_ADDR="2001:db8::1"\nCONFIG_NET_CONFIG_MY_IPV4_ADDR="192.0.2.1"\nCONFIG_NET_CONFIG_MY_IPV4_GW="192.0.2.2"\nCONFIG_DNS_SERVER1="192.0.2.2"\n# DHCP configuration. Until DHCP address is assigned,\n# static configuration above is used instead.\nCONFIG_NET_DHCPV4=y\n\n# \u542f\u7528\u7684 MicroPython \u914d\u7f6e\u9879\uff0c"mpconfigport_full.h" \u8868\u793a\u5168\u91cf\u5f00\u542f\nCONFIG_MICROPY_CONFIGFILE="mpconfigport_full.h"\n')),(0,a.kt)("h2",{id:"2-\u6587\u4ef6\u7cfb\u7edf\u914d\u7f6e"},"2. \u6587\u4ef6\u7cfb\u7edf\u914d\u7f6e"),(0,a.kt)("p",null,"\u5f53\u4f60\u9700\u8981\u5c06 MicroPython \u4ee3\u7801\u7f6e\u4e8e\u6587\u4ef6\u7cfb\u7edf\u4e2d\u65f6\uff0c\u4f60\u9700\u8981\u5bf9\u6587\u4ef6\u7cfb\u7edf\u76f8\u5173\u7684\u914d\u7f6e\u8fdb\u884c\u4fee\u6539\u3002\u5982\u679c\u4f60\u4e0d\u786e\u5b9a\u4f60\u662f\u5426\u9700\u8981\u6b64\u529f\u80fd\uff0c\u8bf7\u53c2\u8003 ",(0,a.kt)("a",{parentName:"p",href:"../bundle_program#which-bundle-way-is-better"},"\u5982\u4f55\u9009\u62e9\u6253\u5305\u65b9\u5f0f")," \u4e2d\u7684\u4ecb\u7ecd\u3002"),(0,a.kt)("p",null,"MicroPython \u5728\u542f\u52a8\u540e\u4f1a\u8bc6\u522b\u662f\u5426\u8bbe\u5907\u6811\u4e2d\u662f\u5426\u5b58\u5728 label \u4e3a ",(0,a.kt)("inlineCode",{parentName:"p"},"storage")," \u7684\u6587\u4ef6\u7cfb\u7edf\u5206\u533a\u914d\u7f6e\uff08\u901a\u8fc7 ",(0,a.kt)("inlineCode",{parentName:"p"},"FLASH_AREA_LABEL_EXISTS(storage)")," \uff09\uff0c\u5982\u679c\u5b58\u5728\uff0c\u5219\u4f1a\u5728 ",(0,a.kt)("inlineCode",{parentName:"p"},"/flash")," \u6302\u8f7d\u70b9\u5c06\u5176\u4f5c\u4e3a\u6587\u4ef6\u7cfb\u7edf\u7684\u6839\u76ee\u5f55\u3002"),(0,a.kt)("p",null,"\u4f46\u5982\u679c\u4f60\u671f\u671b\u5355\u72ec\u5212\u5206\u4e00\u4e2a\u5206\u533a\u4f5c\u4e3a MicroPython \u4ee3\u7801\u76f8\u5173\u6587\u4ef6\u5b58\u653e\u7684\u533a\u57df\uff0c\u4f60\u4e5f\u53ef\u4ee5\u81ea\u884c\u5728\u8bbe\u5907\u6811\u4e2d\u6307\u5b9a\u76f8\u5173\u914d\u7f6e\u540e\uff0c\u5728\u4f60\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"main.c")," \u4e2d\u8c03\u7528\u6302\u8f7d\u521d\u59cb\u5316\u3002"),(0,a.kt)("p",null,"\u4f8b\u5982\u82e5\u4f60\u7684\u8bbe\u5907\u6811\u4e2d\u914d\u7f6e\u4e3a"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-dts"},'/ {\n  chosen {\n    /*\n    * shared memory reserved for the inter-processor communication\n    */\n    zephyr,flash_sysfs_storage = &filesystem_part;\n    zephyr,flash_controller = &flash;\n  };\n\n  fstab {\n    compatible = "zephyr,fstab";\n    lfs1: lfs1 {\n      compatible = "zephyr,fstab,littlefs";\n      mount-point = "/mpy";\n      partition = <&filesystem_part>;\n      automount ;\n      read-size = <16>;\n      prog-size = <16>;\n      cache-size = <64>;\n      lookahead-size = <32>;\n      block-cycles = <512>;\n    };\n  };\n};\n\n&flash0 {\n  partitions {\n    compatible = "fixed-partitions";\n    #address-cells = <1>;\n    #size-cells = <1>;\n    filesystem_part: partition@160000 {\n      label = "mpy"; /* \u6b64\u5904\u4e0b\u6587\u63d0\u53ca\u7684 FLASH_AREA_ID */\n      reg = <0x160000 0x400000>;\n    };\n  };\n};\n')),(0,a.kt)("p",null,"\u4fee\u6539\u4f60\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"{board}.conf")," \u6587\u4ef6\u4e2d\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"CONFIG_MICROPY_CONFIGFILE")," \u503c"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-kconfig"},'CONFIG_MICROPY_CONFIGFILE = "mpconfigport_custom.h"\n')),(0,a.kt)("p",null,"\u800c\u540e\uff0c\u5728\u4f60\u7684\u9879\u76ee\u76ee\u5f55\u4e0b\u65b0\u589e ",(0,a.kt)("inlineCode",{parentName:"p"},"mpconfigport_custom.h")," \uff0c\u5728\u5176\u4e2d\u52a0\u5165\u5982\u4e0b\u5185\u5bb9\uff08\u6ce8\u610f\u4fee\u6539 ",(0,a.kt)("inlineCode",{parentName:"p"},"FLASH_AREA_ID")," \u9644\u5e26\u7684\u53c2\u6570\uff09"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-c"},'#include "mpconfigport_full.h"\n\n#define CUSTOM_VFS_INIT() mp_obj_t bdev = NULL; \\\n    mp_obj_t mount_point; \\\n    const char *mount_point_str = NULL; \\\n    int ret = 0; \\\n    \\\n    mp_obj_t args[] = { MP_OBJ_NEW_SMALL_INT(FLASH_AREA_ID(mpy)), MP_OBJ_NEW_SMALL_INT(4096) }; \\\n    bdev = zephyr_flash_area_type.make_new(&zephyr_flash_area_type, ARRAY_SIZE(args), 0, args); \\\n    mount_point_str = "/flash"; \\\n    \\\n    if ((bdev != NULL)) { \\\n        mount_point = mp_obj_new_str(mount_point_str, strlen(mount_point_str)); \\\n        ret = mp_vfs_mount_and_chdir_protected(bdev, mount_point); \\\n    }\n')),(0,a.kt)("p",null,"\u5bf9 MicroPython \u6765\u8bf4\uff0c\u5f53\u4f60\u6267\u884c ",(0,a.kt)("inlineCode",{parentName:"p"},"import xxx")," \u7684\u65f6\u5019\uff0c\u5b83\u4f1a\u5148\u4ece ",(0,a.kt)("inlineCode",{parentName:"p"},".")," \u76ee\u5f55\u4e2d\u53bb\u67e5\u627e\uff0c\u9ed8\u8ba4\u6307\u5411\u7b2c\u4e00\u4e2a\u6302\u8f7d\u5230 MicroPython \u7684\u6302\u8f7d\u70b9\u3002"),(0,a.kt)("p",null,"\u800c\u540e\uff0c\u5728\u4f60\u7684\u9879\u76ee\u76ee\u5f55\u4e0b\u521b\u5efa ",(0,a.kt)("inlineCode",{parentName:"p"},"py")," \u76ee\u5f55\uff0c\u6267\u884c\u4ee5\u4e0b\u547d\u4ee4\u5c06 ",(0,a.kt)("inlineCode",{parentName:"p"},"py")," \u76ee\u5f55\u4e2d\u7684\u5185\u5bb9\u70e7\u5f55\u5230\u6587\u4ef6\u7cfb\u7edf\u4e2d\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"lisa mpy flash --fs --flash-area=mpy\n")),(0,a.kt)("p",null,"\u6b64\u5904\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"--flash-area=mpy")," \u4f20\u53c2\u542b\u4e49\u4e0e\u4e0a\u6587\u4e2d\u7684\u63d0\u5230\u7684\u3001\u8bbe\u5907\u6811\u4e2d\u914d\u7f6e\u7684\u6587\u4ef6\u7cfb\u7edf\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"label")," \u662f\u540c\u4e00\u4e2a\u4e1c\u897f\u3002\u9ed8\u8ba4\u60c5\u51b5\u4e0b\uff0c label \u8ba4\u4e3a ",(0,a.kt)("inlineCode",{parentName:"p"},"lfs1")," \u3002"),(0,a.kt)("h2",{id:"3-\u542f\u52a8-micropython"},"3. \u542f\u52a8 MicroPython"),(0,a.kt)("p",null,"\u5728\u4f60\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"main.c")," \u4e2d\u7684\u4e3b\u51fd\u6570\u540e\uff0c\u50cf\u4e0b\u8ff0\u65b9\u5f0f\u52a0\u5165\u4e24\u4e2a\u51fd\u6570\u7684\u8c03\u7528\uff0c\u7528\u4e8e\u521d\u59cb\u5316 MicroPython \u5e76\u4e14\u4f7f MicroPython REPL \u53ef\u7528\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-C"},"#include <console/console.h>\n\nint real_main(void);\n\nvoid main(void) {\n    // Other code ...\n\n    console_init();\n\n    real_main();\n}\n")),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"WARNING"),": \u6ce8\u610f\uff0c\u7ecf\u8fc7\u4e0a\u8ff0\u521d\u59cb\u5316\u540e\uff0c\u5c06\u4f7f\u7528\u4e32\u53e3\u7ec8\u7aef\u4f5c\u4e3a MicroPython \u4ea4\u4e92\u5f0f\u7ec8\u7aef\uff0c\u90a3\u4e48 Zephyr \u81ea\u8eab\u7684\u7ec8\u7aef\u5c06\u4e0d\u53ef\u7528\uff08\u5373 ",(0,a.kt)("inlineCode",{parentName:"p"},"CONFIG_SHELL")," \u76f8\u5173\u5185\u5bb9\uff09\u3002"))}m.isMDXComponent=!0}}]);