(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[9782],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return d}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function u(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):u(u({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},_={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),m=l(n),d=a,s=m["".concat(c,".").concat(d)]||m[d]||_[d]||i;return n?r.createElement(s,u(u({ref:t},p),{},{components:n})):r.createElement(s,u({ref:t},p))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,u=new Array(i);u[0]=m;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o.mdxType="string"==typeof e?e:a,u[1]=o;for(var l=2;l<i;l++)u[l]=n[l];return r.createElement.apply(null,u)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},68118:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return p},default:function(){return m}});var r=n(22122),a=n(19756),i=(n(67294),n(3905)),u=["components"],o={},c="\u4e92\u65a5\u91cf",l={unversionedId:"chips/600X/application/kernel/syncmutex",id:"chips/600X/application/kernel/syncmutex",isDocsHomePage:!1,title:"\u4e92\u65a5\u91cf",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/kernel/syncmutex.md",sourceDirName:"chips/600X/application/kernel",slug:"/chips/600X/application/kernel/syncmutex",permalink:"/chips/600X/application/kernel/syncmutex",version:"current",frontMatter:{}},p=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u5e38\u7528API\u63a5\u53e3",id:"\u5e38\u7528api\u63a5\u53e3",children:[]},{value:"\u4e92\u65a5\u91cf\u7684\u4f7f\u7528",id:"\u4e92\u65a5\u91cf\u7684\u4f7f\u7528",children:[]}],_={toc:p};function m(e){var t=e.components,n=(0,a.Z)(e,u);return(0,i.kt)("wrapper",(0,r.Z)({},_,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"\u4e92\u65a5\u91cf"},"\u4e92\u65a5\u91cf"),(0,i.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,i.kt)("p",null,"\u672c\u7ae0\u8282\u5728\u4e0a\u4e2a\u7ae0\u8282-\u540c\u6b65\u4e4b\u4fe1\u53f7\u91cf\u57fa\u7840\u4e0a\u8bb2\u89e3\u4e92\u65a5\u91cf\u7684\u4f7f\u7528\u573a\u666f\u548c\u4f7f\u7528\u65b9\u6cd5\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u5b66\u4e60\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u4e86\u89e3\u5230\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u4e92\u65a5\u91cf\u7684\u57fa\u672c\u4fe1\u606f\u548c\u4f7f\u7528\u573a\u666f"),(0,i.kt)("li",{parentName:"ul"},"\u4e92\u65a5\u91cf\u7684\u4f7f\u7528\u65b9\u6cd5")),(0,i.kt)("p",null,"\u4e92\u65a5\u91cf(Mutex)\u662f\u8868\u73b0\u4e92\u65a5\u73b0\u8c61\u7684\u6570\u636e\u7ed3\u6784\uff0c\u672c\u8d28\u4e0a\u662f\u4e00\u628a\u9501\uff0c\u63d0\u4f9b\u5bf9\u8d44\u6e90\u7684\u72ec\u5360\u8bbf\u95ee\uff0c\u901a\u5e38\u4f5c\u4e3a\u4fdd\u62a4\u591a\u7ebf\u7a0b\u7684\u4e34\u754c\u533a\uff0c\u5f53\u7ebf\u7a0b\u9700\u8981\u8bbf\u95ee\u4e34\u754c\u533a\u65f6\uff0c\u8c03\u7528",(0,i.kt)("inlineCode",{parentName:"p"},"k_mutex_lock"),"\uff0c\u5982\u679c\u5f53\u524d\u4e92\u65a5\u91cf\u662f\u89e3\u9501\u72b6\u6001\uff0c\u5219\u8c03\u7528\u6210\u529f\uff0c\u8c03\u7528\u7ebf\u7a0b\u53ef\u81ea\u7531\u8fdb\u5165\u4e34\u754c\u533a\uff0c\u5982\u679c\u4e92\u65a5\u91cf\u5df2\u7ecf\u67b7\u9501\uff0c\u8c03\u7528\u7ebf\u7a0b\u4f1a\u88ab\u963b\u585e\uff0c\u77e5\u9053\u4e34\u754c\u533a\u4e2d\u6240\u6709\u7684\u7ebf\u7a0b\u5b8c\u6210\u5e76\u8c03\u7528",(0,i.kt)("inlineCode",{parentName:"p"},"k_mutex_unlock"),"\u3002"),(0,i.kt)("p",null,"\u4e92\u65a5\u91cf\u6709\u4ee5\u4e0b\u7279\u6027\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Mutex\u53ea\u80fd\u7528\u4e8e\u7ebf\u7a0b\u4e4b\u95f4\uff0c\u4e0d\u80fd\u7528\u4e8eISR\u4e2d\u3002"),(0,i.kt)("li",{parentName:"ul"},"Mutex unlock\u65f6\u4f1a\u5f15\u53d1\u8c03\u5ea6\u3002"),(0,i.kt)("li",{parentName:"ul"},"\u4f4e\u4f18\u5148\u7ea7\u7ebf\u7a0b\u83b7\u53d6\u9501\u540e\u6709\u9ad8\u4f18\u5148\u7ea7\u7ebf\u7a0b\u7b49\u9501\u65f6\u4f1a\u5f15\u8d77\u4f18\u5148\u7ea7\u5012\u7f6e\u3002")),(0,i.kt)("h2",{id:"\u5e38\u7528api\u63a5\u53e3"},"\u5e38\u7528API\u63a5\u53e3"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"/*\u521d\u59cb\u5316\u4e92\u65a5\u91cf*/\n#define K_MUTEX_DEFINE(name)    \n\n/*\u521d\u59cb\u5316\u4e92\u65a5\u91cf*/\nint k_mutex_init(struct k_mutex * mutex)    \n\n/*\u52a0\u9501\u4e92\u65a5\u91cf*/\nint k_mutex_lock(struct k_mutex * mutex, k_timeout_t timeout)\n\n/*\u89e3\u9501\u4e92\u65a5\u91cf*/\nvoid k_mutex_unlock(struct k_mutex * mutex)\n")),(0,i.kt)("p",null,"\u66f4\u591a",(0,i.kt)("inlineCode",{parentName:"p"},"Mutex API"),"\u63a5\u53e3 \u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,i.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__mutex__apis.html"},"Mutex APIs"),"\u4e2d\u627e\u5230\u3002"),(0,i.kt)("h2",{id:"\u4e92\u65a5\u91cf\u7684\u4f7f\u7528"},"\u4e92\u65a5\u91cf\u7684\u4f7f\u7528"),(0,i.kt)("p",null,"\u4ee5\u4e0b\u662f\u4e92\u65a5\u91cf\u4f7f\u7528\u4f8b\u7a0b\uff0c\u4e24\u4e2a\u7ebf\u7a0b\u90fd\u8981\u8bbf\u95ee\u540c\u4e00\u4e2aIO\u8d44\u6e90\uff0c\u4f46IO\u540c\u65f6\u53ea\u80fd\u88ab\u72ec\u5360\u8bbf\u95ee\uff0c\u56e0\u6b64\u9700\u8981\u4f7f\u7528\u4e92\u65a5\u91cf\u6765\u5b9e\u73b0\uff0c\u5b9e\u73b0\u4ee3\u7801\u5982\u4e0b\uff1a\n",(0,i.kt)("strong",{parentName:"p"},"\u4e92\u65a5\u91cf\u521d\u59cb\u5316"),"\n\u4e00\u4e2a\u4e92\u65a5\u91cf\u4f7f\u7528\u4e00\u4e2a\u7c7b\u578b\u4e3a k_mutex \u7684\u53d8\u91cf\u5b9a\u4e49\uff0c\u6709\u4e24\u79cd\u65b9\u5f0f\u53ef\u4ee5\u5b8c\u6210\u4fe1\u53f7\u91cf\u7684\u521d\u59cb\u5316\uff1a     "),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u65b9\u6cd51\uff0c\u4f7f\u7528\u5b8f\uff1a")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"K_MUTEX_DEFINE(my_mutex);\n")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u65b9\u6cd52\uff0c\u4f7f\u7528\u51fd\u6570 ")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"struct k_mutex my_mutex;\nk_mutex_init(&my_mutex);\n")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u52a0\u9501\u548c\u89e3\u9501\u4fe1\u53f7\u91cf"),"       "),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"#define LED0_NODE_ID DT_NODELABEL(gpioa)\n#define PIN 5\n#define FLAGS 0\n\n#define A_STACK_SIZE 4096\n#define B_STACK_SIZE 4096\n\nK_THREAD_STACK_DEFINE(thread_a_area, A_STACK_SIZE);\nK_THREAD_STACK_DEFINE(thread_b_area, B_STACK_SIZE);\n\nstruct k_thread thread_a_data;\nstruct k_thread thread_b_data;\n\nvoid main(void)\n{\n    ...\n    dev = DEVICE_DT_GET(LED0_NODE_ID);\n    ret = gpio_pin_configure(dev, PIN, GPIO_OUTPUT_ACTIVE | FLAGS);\n    ...\n\n    struct k_mutex my_mutex;\n    k_mutex_init(&my_mutex);\n    ...\n\n    /* create thread_a and thread_b */\n    int pri = k_thread_priority_get(k_current_get());\n    k_thread_create(&thread_a_data, thread_a_area,\n                    K_THREAD_STACK_SIZEOF(thread_a_area),\n                    thread_a, NULL, NULL, NULL, pri, 0, K_NO_WAIT);\n\n    k_thread_create(&thread_b_data, thread_b_area,\n                    K_THREAD_STACK_SIZEOF(thread_b_area), \n                    thread_b, NULL,NULL, NULL, pri, 0, K_NO_WAIT);\n}\n\n/*\u7ebf\u7a0bA\u64cd\u4f5cIO\u65f6\u5bf9\u4e92\u65a5\u91cf\u7684\u64cd\u4f5c*/\nthread_a()\n{\n    k_mutex_lock(&my_mutex, K_FOREVER);\n\n    /*\u64cd\u4f5cGPIO*/\n    gpio_pin_set(dev, PIN, 1);\n\n    k_mutex_unlock(&my_mutex);\n}\n\n/*\u7ebf\u7a0bB\u64cd\u4f5cIO\u65f6\u5bf9\u4e92\u65a5\u91cf\u7684\u64cd\u4f5c*/\nthread_b()\n{\n    k_mutex_lock(&my_mutex, K_FOREVER);\n\n    /*\u64cd\u4f5cGPIO*/\n    gpio_pin_set(dev, PIN, 0);\n\n    k_mutex_unlock(&my_mutex);\n}\n")),(0,i.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},"\u672c\u7ae0\u8282\u540c\u6b65\u4e4b\u4e92\u65a5\u91cf\u5185\u5bb9\u4e0d\u5728",(0,i.kt)("inlineCode",{parentName:"p"},"csk6 sdk"),"\u4e2d\u63d0\u4f9b\u5177\u4f53\u7684\u5b9e\u73b0sample\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u6839\u636e\u4e92\u65a5\u91cf\u7684\u4f7f\u7528\u53c2\u8003\u5c1d\u8bd5\u5b8c\u6210\u5e94\u7528\u5f00\u53d1\u548c\u63a2\u7d22\uff0c\u4ee5\u63d0\u5347\u5f00\u53d1\u719f\u7ec3\u5ea6\u3002"))))}m.isMDXComponent=!0}}]);