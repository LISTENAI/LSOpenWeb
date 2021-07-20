"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireWildcard(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _Link = _interopRequireDefault(require("@docusaurus/Link"));

var _useBaseUrl = _interopRequireDefault(require("@docusaurus/useBaseUrl"));

var _reactRouterDom = require("react-router-dom");

var _themeCommon = require("@docusaurus/theme-common");

var _isInternalUrl = _interopRequireDefault(require("@docusaurus/isInternalUrl"));

require("./styles.css");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }

function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// import IconExternalLink from '@theme/IconExternalLink';
const dropdownLinkActiveClass = 'dropdown__link--active';

function NavLink({
  activeBasePath,
  activeBaseRegex,
  to,
  href,
  label,
  activeClassName = 'subnavbar__link--active',
  prependBaseUrlToHref,
  ...props
}) {
  // TODO all this seems hacky
  // {to: 'version'} should probably be forbidden, in favor of {to: '/version'}
  const toUrl = (0, _useBaseUrl.default)(to);
  const activeBaseUrl = (0, _useBaseUrl.default)(activeBasePath);
  const normalizedHref = (0, _useBaseUrl.default)(href, {
    forcePrependBaseUrl: true
  });
  const isExternalLink = label && href && !(0, _isInternalUrl.default)(href); // const isDropdownLink = activeClassName === dropdownLinkActiveClass;

  return <_Link.default {...href ? {
    href: prependBaseUrlToHref ? normalizedHref : href
  } : {
    isNavLink: true,
    activeClassName,
    to: toUrl,
    ...(activeBasePath || activeBaseRegex ? {
      isActive: (_match, location) => activeBaseRegex ? new RegExp(activeBaseRegex).test(location.pathname) : location.pathname.startsWith(activeBaseUrl)
    } : null)
  }} {...props}>
      {isExternalLink ? <span className="navbar__exlink">
          {label}
          {
        /* <IconExternalLink {...(isDropdownLink && {width: 12, height: 12})} /> */
      }
        </span> : label}
    </_Link.default>;
}

function NavItemDesktop({
  items,
  position,
  className,
  ...props
}) {
  var _props$children;

  const dropdownRef = (0, _react.useRef)(null);
  const dropdownMenuRef = (0, _react.useRef)(null);
  const [showDropdown, setShowDropdown] = (0, _react.useState)(false);
  (0, _react.useEffect)(() => {
    const handleClickOutside = event => {
      if (!dropdownRef.current || dropdownRef.current.contains(event.target)) {
        return;
      }

      setShowDropdown(false);
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, [dropdownRef]);

  const navLinkClassNames = (extraClassName, isDropdownItem = false) => (0, _clsx.default)({
    'subnavbar__item subnavbar__link': !isDropdownItem,
    dropdown__link: isDropdownItem
  }, extraClassName);

  if (!items) {
    return <NavLink className={navLinkClassNames(className)} {...props} />;
  }

  return <div ref={dropdownRef} className={(0, _clsx.default)('navbar__item', 'dropdown', 'dropdown--hoverable', {
    'dropdown--left': position === 'left',
    'dropdown--right': position === 'right',
    'dropdown--show': showDropdown
  })}>
      <NavLink className={navLinkClassNames(className)} {...props} onClick={props.to ? undefined : e => e.preventDefault()} onKeyDown={e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        setShowDropdown(!showDropdown);
      }
    }}>
        {(_props$children = props.children) !== null && _props$children !== void 0 ? _props$children : props.label}
      </NavLink>
      <ul ref={dropdownMenuRef} className="dropdown__menu">
        {items.map(({
        className: childItemClassName,
        ...childItemProps
      }, i) => <li key={i}>
            <NavLink onKeyDown={e => {
          if (i === items.length - 1 && e.key === 'Tab') {
            e.preventDefault();
            setShowDropdown(false);
            const nextNavbarItem = dropdownRef.current.nextElementSibling;

            if (nextNavbarItem) {
              nextNavbarItem.focus();
            }
          }
        }} activeClassName={dropdownLinkActiveClass} className={navLinkClassNames(childItemClassName, true)} {...childItemProps} />
          </li>)}
      </ul>
    </div>;
}

function NavItemMobile({
  items,
  className,
  position: _position,
  // Need to destructure position from props so that it doesn't get passed on.
  ...props
}) {
  var _menuListRef$current, _menuListRef$current2, _props$children2;

  const menuListRef = (0, _react.useRef)(null);
  const {
    pathname
  } = (0, _reactRouterDom.useLocation)();
  const [collapsed, setCollapsed] = (0, _react.useState)(() => {
    var _items$some;

    return (_items$some = !(items !== null && items !== void 0 && items.some(item => (0, _themeCommon.isSamePath)(item.to, pathname)))) !== null && _items$some !== void 0 ? _items$some : true;
  });

  const navLinkClassNames = (extraClassName, isSubList = false) => (0, _clsx.default)('menu__link', {
    'menu__link--sublist': isSubList
  }, extraClassName);

  if (!items) {
    return <li className="menu__list-item">
        <NavLink className={navLinkClassNames(className)} {...props} />
      </li>;
  }

  const menuListHeight = (_menuListRef$current = menuListRef.current) !== null && _menuListRef$current !== void 0 && _menuListRef$current.scrollHeight ? `${(_menuListRef$current2 = menuListRef.current) === null || _menuListRef$current2 === void 0 ? void 0 : _menuListRef$current2.scrollHeight}px` : undefined;
  return <li className={(0, _clsx.default)('menu__list-item', {
    'menu__list-item--collapsed': collapsed
  })}>
      <NavLink role="button" className={navLinkClassNames(className, true)} {...props} onClick={e => {
      e.preventDefault();
      setCollapsed(state => !state);
    }}>
        {(_props$children2 = props.children) !== null && _props$children2 !== void 0 ? _props$children2 : props.label}
      </NavLink>
      <ul className="menu__list" ref={menuListRef} style={{
      height: !collapsed ? menuListHeight : undefined
    }}>
        {items.map(({
        className: childItemClassName,
        ...childItemProps
      }, i) => <li className="menu__list-item" key={i}>
            <NavLink activeClassName="menu__link--active" className={navLinkClassNames(childItemClassName)} {...childItemProps} onClick={props.onClick} />
          </li>)}
      </ul>
    </li>;
}

function SubNavbarItem({
  mobile = false,
  ...props
}) {
  const Comp = mobile ? NavItemMobile : NavItemDesktop;
  return <Comp {...props} />;
}

var _default = SubNavbarItem;
exports.default = _default;