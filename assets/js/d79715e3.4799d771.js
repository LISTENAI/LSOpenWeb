(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[1875],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return u},kt:function(){return _}});var l=t(67294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);n&&(l=l.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,l)}return t}function p(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,l,r=function(e,n){if(null==e)return{};var t,l,r={},a=Object.keys(e);for(l=0;l<a.length;l++)t=a[l],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(l=0;l<a.length;l++)t=a[l],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var c=l.createContext({}),i=function(e){var n=l.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):p(p({},n),e)),t},u=function(e){var n=i(e.components);return l.createElement(c.Provider,{value:n},e.children)},s={inlineCode:"code",wrapper:function(e){var n=e.children;return l.createElement(l.Fragment,{},n)}},d=l.forwardRef((function(e,n){var t=e.components,r=e.mdxType,a=e.originalType,c=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),d=i(t),_=r,L=d["".concat(c,".").concat(_)]||d[_]||s[_]||a;return t?l.createElement(L,p(p({ref:n},u),{},{components:t})):l.createElement(L,p({ref:n},u))}));function _(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var a=t.length,p=new Array(a);p[0]=d;var o={};for(var c in n)hasOwnProperty.call(n,c)&&(o[c]=n[c]);o.originalType=e,o.mdxType="string"==typeof e?e:r,p[1]=o;for(var i=2;i<a;i++)p[i]=t[i];return l.createElement.apply(null,p)}return l.createElement.apply(null,t)}d.displayName="MDXCreateElement"},97656:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return o},contentTitle:function(){return c},metadata:function(){return i},toc:function(){return u},default:function(){return d}});var l=t(22122),r=t(19756),a=(t(67294),t(3905)),p=["components"],o={},c="\u4f7f\u7528LVGL",i={unversionedId:"chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7/use_lvgl",id:"chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7/use_lvgl",isDocsHomePage:!1,title:"\u4f7f\u7528LVGL",description:"\u672c\u8282\u5c06\u4e3b\u8981\u4ecb\u7ecd\u4f7f\u7528\u5728CSK6\u7684Zephyr SDK\u4e2d\uff0c\u5982\u4f55\u4f7f\u7528LVGL\u5bf9\u663e\u793a\u5c4f\u8fdb\u884cGUI\u7684\u8f93\u51fa\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u6211\u4eec\u5c06\u638c\u63e1\uff1a",source:"@site/docs/chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7/use_lvgl.md",sourceDirName:"chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7",slug:"/chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7/use_lvgl",permalink:"/chips/600X/old_doc/\u5e94\u7528\u5f00\u53d1/\u9ad8\u7ea7/use_lvgl",version:"current",frontMatter:{}},u=[{value:"LVGL\u7b80\u4ecb",id:"lvgl\u7b80\u4ecb",children:[]},{value:"LVGL\u5173\u952e\u6982\u5ff5",id:"lvgl\u5173\u952e\u6982\u5ff5",children:[]},{value:"LVGL Sample\u8fd0\u884c",id:"lvgl-sample\u8fd0\u884c",children:[]},{value:"Sample\u4ee3\u7801\u8bf4\u660e",id:"sample\u4ee3\u7801\u8bf4\u660e",children:[]}],s={toc:u};function d(e){var n=e.components,o=(0,r.Z)(e,p);return(0,a.kt)("wrapper",(0,l.Z)({},s,o,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u4f7f\u7528lvgl"},"\u4f7f\u7528LVGL"),(0,a.kt)("p",null,"\u672c\u8282\u5c06\u4e3b\u8981\u4ecb\u7ecd\u4f7f\u7528\u5728CSK6\u7684Zephyr SDK\u4e2d\uff0c\u5982\u4f55\u4f7f\u7528LVGL\u5bf9\u663e\u793a\u5c4f\u8fdb\u884cGUI\u7684\u8f93\u51fa\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u6211\u4eec\u5c06\u638c\u63e1\uff1a"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"\u5bf9LVGL\u7684\u57fa\u672c\u8ba4\u8bc6")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"\u5bf9LVGL\u4e00\u4e9b\u5173\u952e\u6982\u5ff5\u7684\u7406\u89e3")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"\u5728CSK6\u5f00\u53d1\u677f\u4e0a\u8fd0\u884c\u4e00\u4e2aLVGL Sample")),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"NOTE"),": \u672c\u7ae0\u8282\u4f7f\u7528\u7684\u5b9e\u9a8c\u786c\u4ef6\u5e73\u53f0\u4e3a\u3010\u626b\u63cf\u7b14\u5f00\u53d1\u677f\u3011\uff0c\u5bf9\u5e94\u677f\u578b\u4e3a\uff1acsk6012_9s_scanpen"),(0,a.kt)("p",null,"\u672c\u7ae0\u8282\u4f7f\u7528\u7684Sample\u4e3a samples/subsys/display/lvgl"),(0,a.kt)("hr",null),(0,a.kt)("h2",{id:"lvgl\u7b80\u4ecb"},"LVGL\u7b80\u4ecb"),(0,a.kt)("p",null,"LVGL\u5168\u79f0Light and Versatile Graphics Library\uff0c\u662f\u4e00\u4e2a\u5f00\u6e90\u7684GUI\u5e93\uff0c\u5177\u5907\u8d44\u6e90\u6d88\u8017\u5c0f\u3001\u53ef\u79fb\u690d\u5ea6\u9ad8\u7684\u7279\u70b9\uff0c\u56e0\u6b64\u5f88\u9002\u5408\u5728\u5d4c\u5165\u5f0f\u8bbe\u5907\u4e0a\u4f7f\u7528\u3002"),(0,a.kt)("p",null,"\u4f5c\u4e3a\u4e00\u4e2a\u56fe\u5f62\u5e93\uff0cLVGL\u4e5f\u81ea\u5e26\u4e86\u4e30\u5bcc\u7684GUI\u63a7\u4ef6\u53ef\u4ee5\u4f9b\u5f00\u53d1\u8005\u76f4\u63a5\u4f7f\u7528\uff0c\u5982\u5e38\u7528\u7684\u6309\u94ae\uff0c\u56fe\u8868\uff0c\u5217\u8868\uff0c\u6ed1\u5757\u7b49\uff0c\u6b64\u5916\u8fd8\u652f\u6301\u591a\u79cd\u56fe\u5f62\u52a8\u753b\u7684\u8c03\u7528\uff0c\u76f8\u6bd4\u4f20\u7edf\u9a71\u5c4f\u7ed8\u5236GUI\uff0c\u5f00\u53d1\u8005\u4e0d\u9700\u8981\u5927\u91cf\u7684\u4ee3\u7801\u7f16\u5199\u91cf\u548c\u53cd\u590d\u7684\u6548\u679c\u9a8c\u8bc1\uff0c\u5c31\u53ef\u4ee5\u642d\u5efa\u51fa\u4e00\u4e2a\u6bd4\u8f83\u6ee1\u610f\u7684GUI\u3002"),(0,a.kt)("h2",{id:"lvgl\u5173\u952e\u6982\u5ff5"},"LVGL\u5173\u952e\u6982\u5ff5"),(0,a.kt)("p",null,"\u5728\u5f00\u59cbLVGL\u7684Sample\u5b9e\u9a8c\u524d\uff0c\u8ba9\u6211\u4eec\u5148\u7b80\u5355\u4e86\u89e3\u4e00\u4e0bLVGL\u4e00\u4e9b\u6bd4\u8f83\u91cd\u8981\u7684\u6982\u5ff5\uff0c\u65b9\u4fbf\u5bf9\u540e\u7eedSample\u9a8c\u8bc1\u7684\u7406\u89e3\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Display\uff1a")," \u5728LVGL\u4e2d\u4e00\u4e2a\u7269\u7406\u663e\u793a\u5668\u5bf9\u5e94\u4e00\u4e2aDisplay\uff0cLVGL\u652f\u6301\u591a\u4e2a\u7269\u7406\u663e\u793a\u5668\uff0c\u5982\u679c\u5f00\u53d1\u7684\u662f\u50cf\u626b\u63cf\u8bcd\u5178\u7b14\u8fd9\u7c7b\u53ea\u6709\u4e00\u4e2a\u5c4f\u5e55\u8bbe\u5907\uff0c\u5219\u5728\u8fd9\u4e2aLVGL\u5de5\u7a0b\u53ea\u6709\u4e00\u4e2aDisplay\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Screen\uff1a")," \u4e00\u4e2aDisplay\u521b\u5efa\u591a\u4e2aScreen\uff0c\u53ef\u4ee5\u7406\u89e3\u4e3a\u9875\u9762\uff0cScreen\u4e0a\u53ef\u4ee5\u6dfb\u52a0\u5404\u79cdLVGL\u652f\u6301\u7684Widgets(\u7ec4\u4ef6)"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Object\uff1a")," \u662fLVGL\u6784\u6210GUI\u7684\u57fa\u672c\u5355\u4f4d\uff0c\u6bcf\u4e2aWidgets\u90fd\u5c5e\u4e8eObject\uff0c\u4e0a\u9762\u7684Screen(\u9875\u9762)\u4e5f\u662f\u4e00\u4e2aObject\u3002object\u4f7f\u7528\u7236\u5b50\u6811\u5f62\u7ed3\u6784\uff0c\u53ea\u5141\u8bb8\u6709\u4e00\u4e2a\u7236\u8282\u70b9\uff0c\u5141\u8bb8\u6709\u591a\u4e2a\u5b50\u8282\u70b9\u3002screen\u505a\u4e3a\u6839\u8282\u70b9\uff0c\u5141\u8bb8\u6ca1\u6709\u7236\u8282\u70b9\u3002"),(0,a.kt)("hr",null),(0,a.kt)("h2",{id:"lvgl-sample\u8fd0\u884c"},"LVGL Sample\u8fd0\u884c"),(0,a.kt)("p",null,"\u6211\u4eec\u5148\u5728\u6211\u4eec\u7684\u5f00\u53d1\u677f\u4e0a\u8fd0\u884c\u4e00\u4e2aLVGL Sample\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Step 1\uff1a \u521b\u5efaSample")),(0,a.kt)("p",null,"\u4eceSDK\u7684  ",(0,a.kt)("inlineCode",{parentName:"p"},"samples/subsys/display")," \u76ee\u5f55\u4e0b\u62f7\u8d1d ",(0,a.kt)("inlineCode",{parentName:"p"},"lvgl")," \u6587\u4ef6\u5939\u81f3\u4e00\u4e2a\u4f4d\u7f6e\uff0c\u8fd9\u662fSDK\u4e2d\u63d0\u4f9b\u7684\u4e00\u4e2aLVGL Sample\u5de5\u7a0b\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Step 2\uff1a \u4fee\u6539\u914d\u7f6e")),(0,a.kt)("p",null,"\u8fdb\u5165 ",(0,a.kt)("inlineCode",{parentName:"p"},"lvgl")," \u76ee\u5f55\uff0c\u6253\u5f00\u5de5\u7a0b\u914d\u7f6e\u6587\u4ef6 ",(0,a.kt)("inlineCode",{parentName:"p"},"prj.conf")," \uff0c\u5728\u6587\u4ef6\u4e2d\u589e\u52a0\u4ee5\u4e0b\u914d\u7f6e\u9009\u9879\uff1a"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},'# \u5f00\u53d1\u677f\u4f7f\u7528\u7684\u5c4f\u5e55\u662f ST7789V\nCONFIG_LVGL_DISPLAY_DEV_NAME="ST7789V"\n# \u542f\u7528ST7789V\u8bbe\u5907(\u5173\u8054\u8bbe\u5907\u9a71\u52a8\u6587\u4ef6)\nCONFIG_ST7789V=y\n# \u542f\u7528SPI\u9a71\u52a8(\u5c4f\u5e55\u4f7f\u7528SPI\u4f5c\u4e3a\u6570\u636e\u603b\u7ebf)\nCONFIG_SPI=y\n# \u542f\u7528GPIO\u9a71\u52a8(\u5c4f\u5e55\u63a7\u5236\u5f15\u811a\u4f7f\u7528)\nCONFIG_GPIO=y\n')),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Step 3\uff1a \u7f16\u8bd1")),(0,a.kt)("p",null,"\u5728\u5f53\u524d\u5de5\u7a0b\u76ee\u5f55\u4e2d\u6267\u884c ",(0,a.kt)("inlineCode",{parentName:"p"},"lisa zep build -b csk6012_9s_scanpen")," \u6307\u4ee4\uff0c\u6307\u5b9a\u6211\u4eec\u4f7f\u7528\u7684\u5f00\u53d1\u677f\u677f\u578b\u8fdb\u884c\u7f16\u8bd1\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Step 4\uff1a \u70e7\u5f55&\u89c2\u5bdf\u5c4f\u5e55\u8f93\u51fa\u5185\u5bb9")),(0,a.kt)("p",null,"\u7f16\u8bd1\u5b8c\u6210\u540e\uff0c\u6267\u884c ",(0,a.kt)("inlineCode",{parentName:"p"},"lisa zep flash --runner pyocd")," \u6307\u4ee4\u8fdb\u884c\u56fa\u4ef6\u70e7\u5f55"),(0,a.kt)("p",null,"\u70e7\u5f55\u5b8c\u6210\u540e\uff0c\u53ef\u89c2\u5bdf\u5230\u8bbe\u5907\u663e\u793a\u5c4f\u51fa\u73b0\u56fe\u50cf\uff0c\u4e2d\u592e\u6709\u4e00\u4e2atext\u5185\u5bb9\u4e3a\u201dHello world!\u201d\u7684lable\uff0c\u4e0b\u65b9\u4e3a\u4e00\u4e2a\u4e0d\u65ad\u9012\u589e\u7684\u8ba1\u6570\u503c\uff0c\u5982\u56fe\uff1a"),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"image",src:t(81244).Z})),(0,a.kt)("h2",{id:"sample\u4ee3\u7801\u8bf4\u660e"},"Sample\u4ee3\u7801\u8bf4\u660e"),(0,a.kt)("p",null,"LVGL Sample\u7b80\u5355\u5c55\u793a\u4e86object\u7684\u521b\u5efa\u548clable\u63a7\u4ef6\u7684\u4f7f\u7528\uff0c\u4e3b\u8981\u4ee3\u7801\u53ca\u6ce8\u91ca\u5982\u4e0b\uff1a"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-C"},'// \u521b\u5efa\u4e24\u4e2a\u5bf9\u8c61\nlv_obj_t *hello_world_label;\nlv_obj_t *count_label;\n\n\n// \u521b\u5efa\u4e00\u4e2adisplay\u8bbe\u5907\nconst struct device *display_dev;\n\n// \u8fdb\u884c\u8bbe\u5907\u7684\u7ed1\u5b9a\uff0cCONFIG_LVGL_DISPLAY_DEV_NAME\u5df2\u5728prj.conf\u4e2d\u5b9a\u4e49\u4e3aST7789V\ndisplay_dev = device_get_binding(CONFIG_LVGL_DISPLAY_DEV_NAME);\n\n// \u5c06\u521b\u5efa\u7684\u5bf9\u8c61\u505a\u4e3alable\u4f7f\u7528\uff0c\u8bbe\u7f6etext\u5185\u5bb9\u4e3a"Hello world!"\nlv_label_set_text(hello_world_label, "Hello world!");\n// \u5bf9\u8fd9\u4e2alable\u505a\u4f4d\u7f6e\u7684\u6392\u5217\uff0c\u4f4d\u7f6e\u4e3a\u9875\u9762\u6b63\u4e2d\u95f4\uff0c\u540e\u9762\u4e24\u4e2a\u53c2\u6570\u4e3ax/y\u5750\u6807\u504f\u79fb\u91cf\nlv_obj_align(hello_world_label, NULL, LV_ALIGN_CENTER, 0, 0);\n\n// \u5c06count_label\u4f5c\u4e3a\u4e00\u4e2alable\uff0c\u5e76\u5c06\u5176\u7236\u8282\u70b9\u914d\u7f6e\u4e3a\u9ed8\u8ba4\u7684screen\ncount_label = lv_label_create(lv_scr_act(), NULL);\n// \u5bf9\u8fd9\u4e2alable\u505a\u4f4d\u7f6e\u7684\u6392\u5217\uff0c\u4f4d\u7f6e\u4e3a\u5e95\u90e8\u7684\u4e2d\u95f4\nlv_obj_align(count_label, NULL, LV_ALIGN_IN_BOTTOM_MID, 0, 0);\n\n// \u6267\u884c\u4efb\u52a1handler\uff0c\u8fdb\u884c\u5c4f\u5e55\u5185\u5bb9\u5237\u65b0\nlv_task_handler();\ndisplay_blanking_off(display_dev);\n\n// \u4e00\u4e2awhile(1)\u5faa\u73af\u4e2d\uff0c\u6bcf\u4e2a1\u79d2\u505a\u4e00\u6b21count_label\u7684text\u7684\u6570\u503c\u9012\u589e1\nwhile (1) {\n        if ((count % 100) == 0U) {\n                sprintf(count_str, "%d", count/100U);\n                lv_label_set_text(count_label, count_str);\n        }\n        // \u9700\u8981\u5b9a\u671f\u6267\u884clv_task_handler\n        lv_task_handler();\n        k_sleep(K_MSEC(10));\n        ++count;\n}\n')),(0,a.kt)("p",null,"LVGL Sample\u4ec5\u7b80\u5355\u5c55\u793a\u4e86LVGL\u7684GUI\u529f\u80fd\uff0c\u66f4\u591a\u7684\u63a7\u4ef6\u3001\u9ad8\u7ea7\u529f\u80fd\u5e94\u7528\u7528\u6237\u53ef\u4ee5\u81ea\u884c\u9605\u8bfbLVGL\u5e93\u63a5\u53e3\u8bf4\u660e\u6216\u524d\u5f80 ",(0,a.kt)("a",{parentName:"p",href:"https://docs.lvgl.io/7.11/index.html"},"LVGL\u6587\u6863")," \u8fdb\u884c\u4e86\u89e3\u3002"))}d.isMDXComponent=!0},81244:function(e,n,t){"use strict";n.Z=t.p+"assets/images/helloworld-b640100c12f655de64a92e88ada9f239.png"}}]);