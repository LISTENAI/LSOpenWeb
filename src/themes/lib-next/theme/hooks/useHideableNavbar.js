/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import {useState, useCallback, useEffect, useRef} from 'react';
import {useLocation} from 'react-router-dom';
import useScrollPosition from '@theme/hooks/useScrollPosition';
import {useLocationChange} from '@docusaurus/theme-common';

const useHideableNavbar = (hideOnScroll) => {
  const location = useLocation();
  const [isNavbarVisible, setIsNavbarVisible] = useState(hideOnScroll);
  const isFocusedAnchor = useRef(false);
  const [navbarHeight, setNavbarHeight] = useState(0);
  const navbarRef = useCallback((node) => {
    if (node !== null) {
      setNavbarHeight(node.getBoundingClientRect().height);
    }
  }, []);
  useScrollPosition(
    ({scrollY: scrollTop}, {scrollY: lastScrollTop}) => {
      if (!hideOnScroll) {
        return;
      }

      if (scrollTop < navbarHeight) {
        setIsNavbarVisible(true);
        return;
      }

      if (isFocusedAnchor.current) {
        isFocusedAnchor.current = false;
        setIsNavbarVisible(false);
        return;
      }

      if (lastScrollTop && scrollTop === 0) {
        setIsNavbarVisible(true);
      }

      const documentHeight =
        document.documentElement.scrollHeight - navbarHeight;
      const windowHeight = window.innerHeight;

      if (lastScrollTop && scrollTop >= lastScrollTop) {
        setIsNavbarVisible(false);
      } else if (scrollTop + windowHeight < documentHeight) {
        setIsNavbarVisible(true);
      }
    },
    [navbarHeight, isFocusedAnchor],
  );
  useLocationChange((locationChangeEvent) => {
    if (!hideOnScroll || locationChangeEvent.location.hash) {
      return;
    }

    setIsNavbarVisible(true);
  });
  useEffect(() => {
    if (!hideOnScroll) {
      return;
    }

    if (!location.hash) {
      return;
    }

    isFocusedAnchor.current = true;
  }, [location.hash]);
  return {
    navbarRef,
    isNavbarVisible,
  };
};

export default useHideableNavbar;
